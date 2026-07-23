"""
Timestep Forcing DMD Model (with optional Progressive Distillation)

Inherits from DMD and overrides the _initialize_inference_pipeline method,
replacing SelfForcingTrainingPipeline with TimestepForcingTrainingPipeline.

Progressive Distillation (PD) features:
- Uses PD loss (velocity regression / Euler endpoint) for the first pd_num_steps denoising steps
- Subsequent steps continue to use DMD loss
- PD and DMD steps do not overlap
- Supports two computation modes: single_forward (all frames in one forward pass) and rollout (block-by-block KV-cache inference)
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional

from pipeline import TimestepForcingTrainingPipeline
from model.dmd import DMD
from utils.wan_wrapper import WanDiffusionWrapper


def build_bidirectional_cd_sigmas(
    num_timesteps: int, timestep_shift: float
) -> torch.Tensor:
    """FlowMatchScheduler(extra_one_step=True) training sigma grid."""
    if num_timesteps < 2:
        raise ValueError("num_timesteps must be at least 2")
    if timestep_shift <= 0:
        raise ValueError("timestep_shift must be positive")
    base_sigmas = torch.linspace(
        1.0, 0.0, num_timesteps + 1, dtype=torch.float32
    )[:-1]
    return (
        timestep_shift
        * base_sigmas
        / (1.0 + (timestep_shift - 1.0) * base_sigmas)
    )


class TimestepForcingDMD(DMD):
    """
    DMD with Timestep-First rollout and optional Progressive Distillation.

    Differences from the original DMD:
    1. Uses TimestepForcingTrainingPipeline for Timestep-First rollout
    2. Optional Progressive Distillation loss to constrain the first few denoising steps
    """

    def __init__(self, args, device):
        super().__init__(args, device)

        # Online bidirectional consistency distillation replaces the old
        # forward-KL trajectory-pair loss. The frozen 14B real_score supplies
        # one PF-ODE step; student and EMA target use the same trainable causal
        # backbone under a bidirectional full-sequence attention mask.
        self.bidirectional_cd_enabled = bool(
            getattr(args, "bidirectional_cd_enabled", False)
        )
        if self.bidirectional_cd_enabled:
            if getattr(args, "pd_enabled", False):
                raise ValueError(
                    "bidirectional_cd_enabled and pd_enabled are mutually exclusive"
                )
            self.bidirectional_cd_num_timesteps = int(
                getattr(args, "bidirectional_cd_num_timesteps", 48)
            )
            self.bidirectional_cd_timestep_shift = float(
                getattr(args, "bidirectional_cd_timestep_shift", 5.0)
            )
            self.bidirectional_cd_guidance_scale = float(
                getattr(args, "bidirectional_cd_guidance_scale", 3.0)
            )
            self.bidirectional_cd_sft_weight = float(
                getattr(args, "bidirectional_cd_sft_weight", 0.0)
            )
            if self.bidirectional_cd_num_timesteps < 2:
                raise ValueError(
                    "bidirectional_cd_num_timesteps must be at least 2"
                )
            if self.bidirectional_cd_timestep_shift <= 0:
                raise ValueError(
                    "bidirectional_cd_timestep_shift must be positive"
                )
            if self.bidirectional_cd_sft_weight < 0:
                raise ValueError(
                    "bidirectional_cd_sft_weight must be non-negative"
                )
            # Reuse the adjacent-pair grid convention from Causal Forcing++:
            # N nonzero training sigmas come from an N+1 endpoint grid, then
            # randint(0, N-1) selects one of its N-1 adjacent training pairs.
            # The objective here is Bi-CD, not the full Stage-2 method.
            sigmas = build_bidirectional_cd_sigmas(
                self.bidirectional_cd_num_timesteps,
                self.bidirectional_cd_timestep_shift,
            )
            self.register_buffer(
                "bidirectional_cd_sigmas", sigmas, persistent=True
            )
            self.bidirectional_cd_ema.model.num_frame_per_block = (
                self.num_frame_per_block
            )
        
        # ============================================================
        # Progressive Distillation configuration
        # ============================================================
        self.pd_enabled = getattr(args, 'pd_enabled', False)
        self.pd_num_steps = getattr(args, 'pd_num_steps', 0)
        self.pd_loss_weight = getattr(args, 'pd_loss_weight', 1.0)
        self.pd_loss_type = getattr(args, 'pd_loss_type', 'euler_endpoint')
        
        # PD x0 loss target mode (only effective when pd_loss_type='x0'):
        # - 'pred_x0': linearly extrapolate from two adjacent teacher ODE save points to sigma=0 to get the teacher's pred_x0 (default)
        # - 'gt_x0': directly use the ground truth x0 from the teacher ODE's final denoised output (i.e., ode_latent[:, -1])
        self.pd_x0_target_mode = getattr(args, 'pd_x0_target_mode', 'pred_x0')
        assert self.pd_x0_target_mode in ('pred_x0', 'gt_x0'), \
            f"pd_x0_target_mode must be 'pred_x0' or 'gt_x0', got: {self.pd_x0_target_mode}"
        
        # PD computation mode: single_forward is the paper's bidirectional
        # Forward-KL path; rollout remains available for causal ablations.
        self.pd_compute_mode = getattr(args, 'pd_compute_mode', 'single_forward')
        assert self.pd_compute_mode in ('single_forward', 'rollout'), \
            f"pd_compute_mode must be 'single_forward' or 'rollout', got: {self.pd_compute_mode}"
        
        # Dedicated PD rollout pipeline (lazily initialized)
        self.pd_rollout_pipeline: Optional[TimestepForcingTrainingPipeline] = None
        
        if self.pd_enabled and self.pd_num_steps > 0:
            # Forward-KL pairs must be sampled at the student's exact sigma
            # schedule. generate_ode_pairs.py interpolates the teacher ODE path
            # at these values and records them in a required manifest.
            model_timesteps = [
                float(value) for value in self.denoising_step_list
            ]
            if not model_timesteps or model_timesteps[-1] != 0.0:
                model_timesteps.append(0.0)
            self.pd_aligned_sigmas = [
                value / self.num_train_timestep for value in model_timesteps
            ]
            if any(
                left <= right
                for left, right in zip(
                    self.pd_aligned_sigmas, self.pd_aligned_sigmas[1:]
                )
            ):
                raise ValueError(
                    "PD student sigmas must be strictly decreasing, got "
                    f"{self.pd_aligned_sigmas}"
                )
            num_denoising_steps = len(self.pd_aligned_sigmas) - 1
            if self.pd_num_steps > num_denoising_steps:
                raise ValueError(
                    f"pd_num_steps ({self.pd_num_steps}) cannot exceed the "
                    f"number of denoising steps ({num_denoising_steps})"
                )

            self.pd_segments = []
            for i in range(self.pd_num_steps):
                teacher_sigma_input = self.pd_aligned_sigmas[i]
                teacher_sigma_target = self.pd_aligned_sigmas[i + 1]
                input_timestep = teacher_sigma_input * self.num_train_timestep
                target_timestep = teacher_sigma_target * self.num_train_timestep
                self.pd_segments.append({
                    'teacher_input_idx': i,       # Input index in the teacher trajectory
                    'teacher_target_idx': i + 1,  # Target index in the teacher trajectory
                    'input_timestep': input_timestep,
                    'target_timestep': target_timestep,
                    # Precise sigma values of the teacher at these two save points
                    'teacher_sigma_input': teacher_sigma_input,
                    'teacher_sigma_target': teacher_sigma_target,
                })

    def _initialize_inference_pipeline(self):
        """
        Replace SelfForcingTrainingPipeline with TimestepForcingTrainingPipeline.

        Key point: min_exit_step is automatically set to pd_num_steps to ensure PD and DMD do not overlap.
        """
        # If PD is enabled, DMD's min_exit_step is automatically set to pd_num_steps
        # Modification: still follow min_exit_step
        min_exit_step = getattr(self.args, 'min_exit_step', 0)
        
        self.inference_pipeline = TimestepForcingTrainingPipeline(
            denoising_step_list=self.denoising_step_list,
            scheduler=self.scheduler,
            generator=self.generator,
            num_frame_per_block=self.num_frame_per_block,
            independent_first_frame=self.args.independent_first_frame,
            same_step_across_blocks=self.args.same_step_across_blocks,
            last_step_only=self.args.last_step_only,
            num_max_frames=self.num_rollout_frames,
            num_gradient_frames=self.num_training_frames,
            context_noise=self.args.context_noise,
            use_ode_trajectory=getattr(self.args, 'use_ode_trajectory', False),
            always_clean_context=getattr(self.args, 'always_clean_context', False),
            min_exit_step=min_exit_step,
            denoising_order=getattr(self.args, 'denoising_order', 'timestep_first'),
            enable_denoise_cache_fusion=getattr(
                self.args, 'enable_denoise_cache_fusion', True
            ),
        )

    def bidirectional_consistency_loss(
        self,
        clean_latent: torch.Tensor,
        conditional_dict: dict,
        unconditional_dict: dict,
    ) -> Tuple[torch.Tensor, dict]:
        """Online adjacent-step bidirectional consistency loss on real data."""
        if not self.bidirectional_cd_enabled:
            return clean_latent.new_zeros(()), {}
        if clean_latent.ndim != 5:
            raise ValueError(
                "clean_latent must have shape [B,F,C,H,W], got "
                f"{tuple(clean_latent.shape)}"
            )
        if not torch.isfinite(clean_latent).all():
            raise ValueError("clean_latent contains NaN or Inf")

        pair_index = torch.randint(
            0,
            self.bidirectional_cd_num_timesteps - 1,
            (1,),
            device=clean_latent.device,
        )
        if torch.distributed.is_initialized():
            torch.distributed.broadcast(pair_index, src=0)
        cd_sigmas = self.bidirectional_cd_sigmas.to(clean_latent.device)
        sigma = cd_sigmas[pair_index]
        sigma_next = cd_sigmas[pair_index + 1]
        sigma_view = sigma.reshape(1, 1, 1, 1, 1)
        sigma_next_view = sigma_next.reshape(1, 1, 1, 1, 1)
        noise = torch.randn_like(clean_latent)
        noisy = (
            (1.0 - sigma_view) * clean_latent.float()
            + sigma_view * noise.float()
        ).to(self.dtype)
        batch_size, num_frames = clean_latent.shape[:2]
        timestep = (sigma * self.num_train_timestep).expand(
            batch_size, num_frames
        ).to(dtype=torch.float32)
        next_timestep = (sigma_next * self.num_train_timestep).expand(
            batch_size, num_frames
        ).to(dtype=torch.float32)

        with torch.no_grad():
            teacher_flow_cond, _ = self.real_score(
                noisy_image_or_video=noisy,
                conditional_dict=conditional_dict,
                timestep=timestep,
                return_x0=False,
            )
            teacher_flow_uncond, _ = self.real_score(
                noisy_image_or_video=noisy,
                conditional_dict=unconditional_dict,
                timestep=timestep,
                return_x0=False,
            )
            # Standard CFG. Do not reuse DMD's cond+s*(cond-uncond), whose
            # effective guidance scale is 1+s.
            teacher_flow = teacher_flow_uncond + (
                self.bidirectional_cd_guidance_scale
                * (teacher_flow_cond - teacher_flow_uncond)
            )
            next_noisy = (
                noisy.float()
                + (sigma_next_view - sigma_view) * teacher_flow.float()
            ).to(self.dtype)
            ema_flow, _ = self.bidirectional_cd_ema(
                noisy_image_or_video=next_noisy,
                conditional_dict=conditional_dict,
                timestep=next_timestep,
                attention_mode="bidirectional",
                return_x0=False,
            )
            target = (
                next_noisy.float() - sigma_next_view * ema_flow.float()
            )

        student_flow, _ = self.generator(
            noisy_image_or_video=noisy,
            conditional_dict=conditional_dict,
            timestep=timestep,
            attention_mode="bidirectional",
            return_x0=False,
        )
        prediction = noisy.float() - sigma_view * student_flow.float()
        cd_loss = F.mse_loss(prediction, target.detach(), reduction="mean")
        if self.bidirectional_cd_sft_weight > 0:
            sft_loss = F.mse_loss(
                prediction, clean_latent.float(), reduction="mean"
            )
        else:
            sft_loss = cd_loss.new_zeros(())
        total = (
            cd_loss + self.bidirectional_cd_sft_weight * sft_loss
        )
        return total, {
            "bidirectional_cd_loss_raw": cd_loss.detach(),
            "bidirectional_cd_sft_loss": sft_loss.detach(),
            "bidirectional_cd_pair_index": pair_index.detach().float(),
            "bidirectional_cd_sigma": sigma.detach(),
            "bidirectional_cd_sigma_next": sigma_next.detach(),
            "bidirectional_cd_prediction_norm": (
                prediction.square().mean().sqrt().detach()
            ),
            "bidirectional_cd_target_norm": (
                target.square().mean().sqrt().detach()
            ),
        }

    # ================================================================
    # PD Rollout related methods
    # ================================================================

    def _initialize_pd_rollout_pipeline(self, input_timestep: torch.Tensor):
        """
        Lazily initialize the dedicated PD rollout TimestepForcingTrainingPipeline.

        Key differences from DMD's inference_pipeline:
        - denoising_step_list only contains the current PD step's input_timestep (single step)
        - last_step_only=True, ensuring exit at the only step
        - min_exit_step=0, since there is only one step, it must exit at step 0
        - Other parameters (num_frame_per_block, use_ode_trajectory, always_clean_context, etc.)
          remain consistent with the DMD pipeline to ensure aligned inference behavior

        Args:
            input_timestep: The input timestep of the current PD step (e.g., 1000, 750, etc.)
        """
        self.pd_rollout_pipeline = TimestepForcingTrainingPipeline(
            denoising_step_list=input_timestep.unsqueeze(0),  # Keep the original dtype after warp, avoid float->long truncation
            scheduler=self.scheduler,
            generator=self.generator,
            num_frame_per_block=self.num_frame_per_block,
            independent_first_frame=False,  # PD data has no independent first frame
            same_step_across_blocks=True,   # Must be True for Timestep-First mode
            last_step_only=True,            # Single-step pipeline, must exit at the last (only) step
            num_max_frames=self.num_training_frames,  # PD data frame count = training frame count, no need for a larger KV cache
            num_gradient_frames=self.num_training_frames,
            context_noise=self.args.context_noise,
            use_ode_trajectory=getattr(self.args, 'use_ode_trajectory', False),
            always_clean_context=getattr(self.args, 'always_clean_context', False),
            min_exit_step=0,                # Single-step pipeline, do not skip any step
            enable_denoise_cache_fusion=getattr(
                self.args, 'enable_denoise_cache_fusion', True
            ),
        )

    def _pd_rollout(
        self,
        teacher_input: torch.Tensor,
        input_timestep: torch.Tensor,
        conditional_dict: dict,
    ) -> torch.Tensor:
        """
        Execute block-by-block rollout starting from a teacher save point.

        Core workflow:
        1. Create a pipeline with denoising_step_list=[input_timestep] (single step)
        2. Pass teacher_input as the noise parameter to inference_with_trajectory
        3. The pipeline executes the only step: block-by-block KV-cache-based forward, then exits
        4. Return the denoised output (x0_pred) of all blocks

        Differences from DMD rollout:
        - DMD rollout starts from pure noise and undergoes multi-step denoising
        - PD rollout starts from a teacher save point and performs only one denoising step
        - Both share the same block-by-block KV-cache inference logic

        Args:
            teacher_input: A save point from the teacher trajectory [B, F, C, H, W]
            input_timestep: The timestep of the current PD step (scalar tensor, e.g., 1000)
            conditional_dict: Conditioning information dict

        Returns:
            denoised_output: Denoised latent from the rollout [B, F, C, H, W]
        """
        # Lazily initialize or update the pipeline's denoising_step_list
        # Each call must ensure denoising_step_list matches the current input_timestep
        if self.pd_rollout_pipeline is None:
            self._initialize_pd_rollout_pipeline(input_timestep)
        else:
            # Update denoising_step_list (avoid recreating the pipeline object)
            self.pd_rollout_pipeline.denoising_step_list = input_timestep.unsqueeze(0)  # Keep original dtype
        
        # Execute rollout
        # teacher_input is passed as the noise parameter; the pipeline treats it as the noisy latent at the current timestep
        # initial_latent=None: PD data does not need I2V conditioning
        output, _, _ = self.pd_rollout_pipeline.inference_with_trajectory(
            noise=teacher_input.to(self.dtype),
            initial_latent=None,
            **conditional_dict,
        )
        
        # output shape: [B, F, C, H, W] (since initial_latent=None, output frame count == input frame count)
        return output

    # ================================================================
    # Progressive Distillation Loss
    # ================================================================

    def progressive_distillation_loss(
        self,
        ode_latent: torch.Tensor,
        conditional_dict: dict,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute Progressive Distillation loss.

        Supports two computation modes:
        - single_forward: All frames in one forward pass (blockwise causal attention)
        - rollout: Block-by-block KV-cache-based inference (aligned with inference behavior)

        For each step in pd_num_steps, adjacent point pairs are taken from the teacher trajectory,
        and the student is trained to predict results that approximate the teacher's next point
        using the teacher's input point as input.

        Args:
            ode_latent: Teacher ODE trajectory [B, num_saves, F, C, H, W]
                        Arranged by save_indices, from noisy to clean
            conditional_dict: Conditioning information dict

        Returns:
            pd_loss: PD loss scalar
            pd_log_dict: Logging information
        """
        if not self.pd_enabled or self.pd_num_steps == 0:
            return torch.tensor(0.0, device=self.device), {}
        
        # Randomly select one PD step for training (similar to ODE regression strategy, reduces memory usage)
        pd_step_idx = torch.randint(0, self.pd_num_steps, (1,), device=self.device)
        # Broadcast to synchronize, ensuring all ranks use the same step
        if torch.distributed.is_initialized():
            torch.distributed.broadcast(pd_step_idx, src=0)
        pd_step_idx = pd_step_idx.item()
        
        segment = self.pd_segments[pd_step_idx]
        
        # Extract input and target from the teacher trajectory
        # ode_latent: [B, num_saves, F, C, H, W]
        teacher_input = ode_latent[:, segment['teacher_input_idx']]    # [B, F, C, H, W]
        teacher_target = ode_latent[:, segment['teacher_target_idx']]  # [B, F, C, H, W]
        
        input_timestep = torch.as_tensor(
            segment['input_timestep'], device=self.device, dtype=torch.float32
        ).reshape(())
        target_timestep = segment['target_timestep']
        
        # Get student prediction based on computation mode
        if self.pd_compute_mode == 'single_forward':
            flow_pred, x0_pred = self._pd_single_forward(
                teacher_input, input_timestep, conditional_dict)
        elif self.pd_compute_mode == 'rollout':
            flow_pred, x0_pred = self._pd_rollout_forward(
                teacher_input, input_timestep, conditional_dict)
        else:
            raise ValueError(f"Unknown pd_compute_mode: {self.pd_compute_mode}")
        
        # Get GT x0 (teacher ODE final denoised output, i.e., the last save point in the trajectory)
        gt_x0 = ode_latent[:, -1]  # [B, F, C, H, W]
        
        # Compute loss
        pd_loss, log_dict = self._compute_pd_loss(
            flow_pred=flow_pred,
            x0_pred=x0_pred,
            teacher_input=teacher_input,
            teacher_target=teacher_target,
            segment=segment,
            gt_x0=gt_x0,
        )
        
        log_dict.update({
            'pd_loss_raw': pd_loss.detach(),
            'pd_step_idx': torch.tensor(pd_step_idx, device=self.device).float(),
            'pd_compute_mode': torch.tensor(
                0.0 if self.pd_compute_mode == 'single_forward' else 1.0,
                device=self.device),
            'pd_input_timestep': torch.tensor(input_timestep, device=self.device).float() 
                if isinstance(input_timestep, (int, float)) 
                else input_timestep.float(),
        })
        
        return pd_loss, log_dict

    def _pd_single_forward(
        self,
        teacher_input: torch.Tensor,
        input_timestep,
        conditional_dict: dict,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        PD single_forward mode: all frames in one bidirectional-attention pass.

        Forward-KL must not modify the causal DMD rollout path. The same trainable
        generator weights are evaluated with a full bidirectional mask here only.

        Args:
            teacher_input: [B, F, C, H, W]
            input_timestep: The timestep of the current step (scalar or tensor)
            conditional_dict: Conditioning information

        Returns:
            flow_pred: [B, F, C, H, W]
            x0_pred: [B, F, C, H, W]
        """
        batch_size, num_frames = teacher_input.shape[:2]
        
        timestep = torch.full(
            [batch_size, num_frames],
            float(input_timestep),
            device=self.device,
            dtype=torch.float32,
        )
        
        flow_pred, x0_pred = self.generator(
            noisy_image_or_video=teacher_input.to(self.dtype),
            conditional_dict=conditional_dict,
            timestep=timestep,
            attention_mode="bidirectional",
        )
        
        return flow_pred, x0_pred

    def _pd_rollout_forward(
        self,
        teacher_input: torch.Tensor,
        input_timestep,
        conditional_dict: dict,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        PD rollout mode: Block-by-block KV-cache-based inference.

        Fully aligned with inference behavior: blocks are input sequentially, using KV cache
        to carry historical information, where earlier blocks' information flows to later blocks
        via KV cache.

        Note: The rollout returns denoised output (x0_pred); flow_pred needs to be derived
        from x0_pred (used for velocity/euler_endpoint loss types).

        Args:
            teacher_input: [B, F, C, H, W]
            input_timestep: The timestep of the current step (scalar or tensor)
            conditional_dict: Conditioning information

        Returns:
            flow_pred: [B, F, C, H, W] (derived from x0_pred)
            x0_pred: [B, F, C, H, W] (denoised output from rollout)
        """
        batch_size, num_frames = teacher_input.shape[:2]
        
        # Execute rollout to get denoised output
        x0_pred = self._pd_rollout(
            teacher_input=teacher_input,
            input_timestep=input_timestep,
            conditional_dict=conditional_dict,
        )
        
        # Derive flow_pred from x0_pred (used for velocity/euler_endpoint loss types)
        timestep = torch.full(
            [batch_size, num_frames],
            float(input_timestep),
            device=self.device,
            dtype=torch.float32,
        )
        
        flow_pred = WanDiffusionWrapper._convert_x0_to_flow_pred(
            scheduler=self.scheduler,
            x0_pred=x0_pred.flatten(0, 1),
            xt=teacher_input.to(self.dtype).flatten(0, 1),
            timestep=timestep.flatten(0, 1),
        ).unflatten(0, (batch_size, num_frames))
        
        return flow_pred, x0_pred

    def _compute_pd_loss(
        self,
        flow_pred: torch.Tensor,
        x0_pred: torch.Tensor,
        teacher_input: torch.Tensor,
        teacher_target: torch.Tensor,
        segment: dict,
        gt_x0: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute PD loss based on pd_loss_type.

        This method is independent of the computation mode (single_forward/rollout)
        and only depends on the student's predictions and the teacher's targets.

        Args:
            flow_pred: Student's flow prediction [B, F, C, H, W]
            x0_pred: Student's x0 prediction [B, F, C, H, W]
            teacher_input: Input point from the teacher trajectory [B, F, C, H, W]
            teacher_target: Target point from the teacher trajectory [B, F, C, H, W]
            segment: PD segment information dict
            gt_x0: Ground truth x0 from teacher ODE final denoised output [B, F, C, H, W]
                   (only used when pd_x0_target_mode='gt_x0')

        Returns:
            pd_loss: Loss scalar
            log_dict: Logging information
        """
        batch_size, num_frames = teacher_input.shape[:2]
        input_timestep = segment['input_timestep']
        target_timestep = segment['target_timestep']
        log_dict = {}
        teacher_sigma_input = float(segment['teacher_sigma_input'])
        teacher_sigma_target = float(segment['teacher_sigma_target'])
        delta_sigma = teacher_sigma_target - teacher_sigma_input
        if abs(delta_sigma) < 1.0e-12:
            raise ValueError(
                "PD teacher segment has zero sigma span: "
                f"{teacher_sigma_input} -> {teacher_sigma_target}"
            )
        
        if self.pd_loss_type == 'euler_endpoint':
            # Step over the teacher segment's real sigma interval. Using the
            # student's nominal four-step interval is wrong when shifts differ.
            euler_result = teacher_input.float() + flow_pred.float() * delta_sigma
            
            pd_loss = F.mse_loss(euler_result.float(), teacher_target.float())
            log_dict['pd_euler_loss'] = pd_loss.detach()
            
        elif self.pd_loss_type == 'velocity':
            # Regress the chord velocity using the teacher's exact saved sigmas.
            # This also avoids the old 0-D target_timestep.unsqueeze(1) crash.
            target_velocity = (
                (teacher_target.double() - teacher_input.double()) / delta_sigma
            ).float()
            
            pd_loss = F.mse_loss(flow_pred.float(), target_velocity)
            log_dict['pd_velocity_loss'] = pd_loss.detach()
            
        elif self.pd_loss_type == 'x0':
            # ============================================================
            # X0 Regression: student's x0_pred approximates teacher's x0 target
            #
            # Two target modes are supported (controlled by pd_x0_target_mode):
            #
            # 1. pred_x0 (default):
            #    Derive teacher's pred_x0 from two save points on the teacher ODE path:
            #      v_teacher = (x_{t'} - x_t) / (sigma_{t'} - sigma_t)    (chord flow)
            #      x0_teacher = x_t - sigma_t * v_teacher                   (extrapolate to sigma=0)
            #    Geometric meaning: In (sigma, x) space, two save points define a chord;
            #                       extending it to sigma=0 gives the teacher's estimate of x0.
            #
            # 2. gt_x0:
            #    Directly use the ground truth x0 from the teacher ODE's final denoised output
            #    (ode_latent[:, -1]).
            #    This is the final clean latent after the teacher's full 48-step ODE sampling,
            #    representing the teacher's final and most accurate estimate of x0.
            #    All PD steps share the same gt_x0 target, providing a more direct signal.
            # ============================================================
            
            if self.pd_x0_target_mode == 'gt_x0':
                # GT X0 mode: directly use teacher ODE final denoised output
                assert gt_x0 is not None, \
                    "pd_x0_target_mode='gt_x0' requires gt_x0 (ode_latent[:, -1]) to be provided"
                teacher_x0 = gt_x0.float()
                
                pd_loss = F.mse_loss(x0_pred.float(), teacher_x0.float())
                log_dict['pd_x0_loss'] = pd_loss.detach()
                log_dict['pd_x0_target_mode'] = torch.tensor(1.0, device=self.device)  # 1.0 = gt_x0
            else:
                # pred_x0 mode (default): linearly extrapolate from two save points
                sigma_t = teacher_sigma_input
                sigma_t_next = teacher_sigma_target
                delta_sigma = sigma_t_next - sigma_t  # Negative (from noisy to clean)
                
                # Compute teacher's pred_x0 in double precision to avoid numerical errors
                teacher_x0 = (
                    teacher_input.double() 
                    - sigma_t * (teacher_target.double() - teacher_input.double()) / delta_sigma
                ).float()
                
                pd_loss = F.mse_loss(x0_pred.float(), teacher_x0.float())
                log_dict['pd_x0_loss'] = pd_loss.detach()
                log_dict['pd_teacher_sigma_input'] = torch.tensor(sigma_t, device=self.device).float()
                log_dict['pd_teacher_sigma_target'] = torch.tensor(sigma_t_next, device=self.device).float()
                log_dict['pd_x0_target_mode'] = torch.tensor(0.0, device=self.device)  # 0.0 = pred_x0
            
        else:
            raise ValueError(f"Unknown pd_loss_type: {self.pd_loss_type}")
        
        return pd_loss, log_dict
