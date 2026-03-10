"""
Timestep Forcing Training Pipeline

Core idea: swap the dimension order of the denoising loop
- Original Self-Forcing: outer loop Frame, inner loop Timestep
- Timestep Forcing: outer loop Timestep, inner loop Frame

Advantages:
1. Context is in a noisy (uncertain) state, leveraging diffusion's self-correction ability
2. Each timestep has a correction opportunity, so errors do not fully propagate to subsequent chunks
3. Training-inference consistency: noisy context is seen during training, and the same strategy is used during inference
"""

from utils.wan_wrapper import WanDiffusionWrapper
from utils.scheduler import SchedulerInterface
from typing import List, Optional, Tuple
import torch
import torch.distributed as dist


class TimestepForcingTrainingPipeline:
    """
    Timestep-First rollout pipeline for training.

    Key differences from SelfForcingTrainingPipeline:
    1. The outer loop is over timesteps, and the inner loop is over frames
    2. KV cache is reset and re-cached with initial_latent at the start of each timestep
    3. Context uses the noisy result at the current timestep (or the result noised to the next step), instead of clean
    """
    
    def __init__(self,
                 denoising_step_list: List[int],
                 scheduler: SchedulerInterface,
                 generator: WanDiffusionWrapper,
                 num_frame_per_block: int = 3,
                 independent_first_frame: bool = False,
                 same_step_across_blocks: bool = False,
                 last_step_only: bool = False,
                 num_max_frames: int = 21,
                 num_gradient_frames: int = 21,
                 context_noise: int = 0,
                 use_ode_trajectory: bool = False,
                 always_clean_context: bool = False,
                 min_exit_step: int = 0,
                 denoising_order: str = "timestep_first",
                 **kwargs):
        """
        Args:
            denoising_step_list: List of denoising steps, e.g. [1000, 750, 500, 250]
            scheduler: Scheduler
            generator: Generator (Wan Diffusion Wrapper)
            num_frame_per_block: Number of frames per block
            independent_first_frame: Whether the first frame is independent
            same_step_across_blocks: Whether all blocks use the same randomly sampled step
            last_step_only: Whether to compute gradients only at the last step
            num_max_frames: Maximum number of frames (used for KV cache initialization, equals rollout frame count)
            num_gradient_frames: Number of frames for gradient computation (e.g. 21), gradients are only computed for the last this many frames
            context_noise: Noise level for clean context
            use_ode_trajectory: Whether to use ODE trajectory (deterministic) instead of SDE trajectory (stochastic noising)
                - False (default): Uses add_noise + new random noise (SDE style, consistent with original)
                - True: Uses scheduler.step (ODE style, more consistent between training and inference)
            always_clean_context: Whether to always use clean context to update KV cache
                - False (default): Intermediate denoising steps use noisy context (latent noised to the next step)
                - True: All steps use clean context (denoised_pred), noisy context never appears
            min_exit_step: Minimum index when randomly sampling exit steps (default 0)
                - 0 (default): All denoising steps can be sampled
                - 1: Skip step 0 (i.e., do not train on step 0)
                - 2: Skip the first two steps (neither step 0 nor step 1 is trained)
                - Used to control skipping the first few denoising steps with high noise levels
            denoising_order: Dimension order of the denoising loop
                - "timestep_first" (default): outer loop timestep, inner loop frame block
                - "frame_first": outer loop frame block, inner loop timestep
        """
        super().__init__()
        self.scheduler = scheduler
        self.generator = generator
        self.denoising_step_list = denoising_step_list
        if self.denoising_step_list[-1] == 0:
            self.denoising_step_list = self.denoising_step_list[:-1]

        # Wan specific hyperparameters
        model = self.generator.model
        self.num_transformer_blocks = len(model.blocks)
        self.num_attention_heads = model.num_heads
        self.attention_head_dim = model.dim // model.num_heads
        self.text_context_len = model.text_len
        self.patch_size = model.patch_size
        self.frame_seq_length = None
        self.num_frame_per_block = num_frame_per_block
        self.context_noise = context_noise
        self.i2v = False

        self.kv_cache1 = None
        self.kv_cache2 = None
        self.independent_first_frame = independent_first_frame
        self.same_step_across_blocks = same_step_across_blocks
        self.last_step_only = last_step_only
        self.max_cache_frames = num_max_frames
        self.kv_cache_size = None
        self.num_gradient_frames = num_gradient_frames
        self.use_ode_trajectory = use_ode_trajectory
        self.always_clean_context = always_clean_context
        self.min_exit_step = min_exit_step
        assert denoising_order in ("timestep_first", "frame_first"), \
            f"denoising_order must be 'timestep_first' or 'frame_first', got: {denoising_order}"
        self.denoising_order = denoising_order

    def _update_runtime_cache_spec(self, height: int, width: int):
        patch_h = self.patch_size[1]
        patch_w = self.patch_size[2]
        if height % patch_h != 0 or width % patch_w != 0:
            raise ValueError(
                f"Latent spatial size ({height}, {width}) is incompatible with patch size {self.patch_size}"
            )
        self.frame_seq_length = (height // patch_h) * (width // patch_w)
        self.kv_cache_size = self.max_cache_frames * self.frame_seq_length

    def generate_and_sync_list(self, num_blocks: int, num_denoising_steps: int, device: torch.device) -> List[int]:
        """
        Generate and synchronize a list of randomly selected exit steps (aligned with SelfForcingTrainingPipeline).

        Args:
            num_blocks: Number of blocks
            num_denoising_steps: Number of denoising steps
            device: Device

        Returns:
            exit_flags: List of exit step indices for each block
        """
        rank = dist.get_rank() if dist.is_initialized() else 0

        if rank == 0:
            # Generate random indices
            indices = torch.randint(
                low=self.min_exit_step,
                high=num_denoising_steps,
                size=(num_blocks,),
                device=device
            )
            if self.last_step_only:
                indices = torch.ones_like(indices) * (num_denoising_steps - 1)
        else:
            indices = torch.empty(num_blocks, dtype=torch.long, device=device)

        dist.broadcast(indices, src=0)  # Broadcast the random indices to all ranks
        return indices.tolist()

    def inference_with_trajectory(
            self,
            noise: torch.Tensor,
            initial_latent: Optional[torch.Tensor] = None,
            return_sim_step: bool = False,
            **conditional_dict
    ) -> Tuple[torch.Tensor, Optional[int], Optional[int]]:
        """
        Execute Timestep-First rollout and return the generated result.

        Core difference:
        - Original: for block in blocks: for step in steps: denoise()
        - This method: for step in steps: for block in blocks: denoise()

        Args:
            noise: Input noise [B, F, C, H, W]
            initial_latent: Initial latent (for I2V)
            return_sim_step: Whether to return sampling step information
            **conditional_dict: Conditioning information (text embeddings, etc.)

        Returns:
            output: Generated latent [B, F_out, C, H, W]
            denoised_timestep_from: Denoising start timestep
            denoised_timestep_to: Denoising end timestep
        """
        batch_size, num_frames, num_channels, height, width = noise.shape
        self._update_runtime_cache_spec(height, width)
        
        # Compute number of blocks
        if not self.independent_first_frame or (self.independent_first_frame and initial_latent is not None):
            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block
        else:
            assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block
        
        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames
        num_denoising_steps = len(self.denoising_step_list)
        
        # Initialize output
        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype
        )

        # Step 1: Initialize KV cache to all zeros
        self._initialize_kv_cache(
            batch_size=batch_size, dtype=noise.dtype, device=noise.device
        )
        self._initialize_crossattn_cache(
            batch_size=batch_size, dtype=noise.dtype, device=noise.device
        )

        # Step 2: Cache context feature
        current_start_frame = 0
        if initial_latent is not None:
            timestep = torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0
            output[:, :1] = initial_latent
            with torch.no_grad():
                self.generator(
                    noisy_image_or_video=initial_latent,
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length
                )
            current_start_frame += 1

        # Step 3: Prepare block info
        all_num_frames = [self.num_frame_per_block] * num_blocks
        if self.independent_first_frame and initial_latent is None:
            all_num_frames = [1] + all_num_frames
        
        # Compute the start frame index for each block
        block_start_frames = []
        temp_start = current_start_frame
        for nf in all_num_frames:
            block_start_frames.append(temp_start)
            temp_start += nf
        
        # Initialize noisy input for each block
        noisy_inputs = []
        for i, (start_frame, nf) in enumerate(zip(block_start_frames, all_num_frames)):
            noisy_input = noise[:, start_frame - num_input_frames:start_frame + nf - num_input_frames]
            noisy_inputs.append(noisy_input)
        
        # Generate exit step list (aligned with SelfForcingTrainingPipeline)
        exit_flags = self.generate_and_sync_list(len(all_num_frames), num_denoising_steps, device=noise.device)
        start_gradient_frame_index = num_output_frames - self.num_gradient_frames

        # ========== Dispatch to different loops based on denoising_order ==========
        if self.denoising_order == "timestep_first":
            unified_exit_step = self._timestep_first_loop(
                batch_size=batch_size,
                noise=noise,
                output=output,
                noisy_inputs=noisy_inputs,
                block_start_frames=block_start_frames,
                all_num_frames=all_num_frames,
                exit_flags=exit_flags,
                num_denoising_steps=num_denoising_steps,
                start_gradient_frame_index=start_gradient_frame_index,
                initial_latent=initial_latent,
                conditional_dict=conditional_dict,
            )
        else:
            unified_exit_step = self._frame_first_loop(
                batch_size=batch_size,
                noise=noise,
                output=output,
                noisy_inputs=noisy_inputs,
                block_start_frames=block_start_frames,
                all_num_frames=all_num_frames,
                exit_flags=exit_flags,
                num_denoising_steps=num_denoising_steps,
                start_gradient_frame_index=start_gradient_frame_index,
                conditional_dict=conditional_dict,
            )

        # Compute denoised_timestep_from and denoised_timestep_to (aligned with SelfForcingTrainingPipeline)
        # Note: need to move denoising_step_list elements to cuda to match scheduler.timesteps
        if not self.same_step_across_blocks:
            denoised_timestep_from, denoised_timestep_to = None, None
        elif unified_exit_step == len(self.denoising_step_list) - 1:
            denoised_timestep_to = 0
            denoised_timestep_from = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[unified_exit_step].cuda()).abs(), dim=0).item()
        else:
            denoised_timestep_to = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[unified_exit_step + 1].cuda()).abs(), dim=0).item()
            denoised_timestep_from = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[unified_exit_step].cuda()).abs(), dim=0).item()

        if return_sim_step:
            return output, denoised_timestep_from, denoised_timestep_to, unified_exit_step + 1

        return output, denoised_timestep_from, denoised_timestep_to

    # ================================================================
    # Noising helper method: unified noising logic for ODE/SDE trajectories
    # ================================================================

    def _compute_noisy_next(
        self,
        flow_pred: torch.Tensor,
        denoised_pred: torch.Tensor,
        noisy_input: torch.Tensor,
        timestep: torch.Tensor,
        next_timestep,
        batch_size: int,
        current_num_frames: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Compute the result noised to next_timestep (ODE or SDE trajectory)."""
        if self.use_ode_trajectory:
            return self.scheduler.step(
                flow_pred.flatten(0, 1),
                timestep.flatten(0, 1),
                noisy_input.flatten(0, 1),
                target_timestep=next_timestep
            ).unflatten(0, denoised_pred.shape[:2])
        else:
            return self.scheduler.add_noise(
                denoised_pred.flatten(0, 1),
                torch.randn_like(denoised_pred.flatten(0, 1)),
                next_timestep * torch.ones(
                    [batch_size * current_num_frames], device=device, dtype=torch.long)
            ).unflatten(0, denoised_pred.shape[:2])

    # ================================================================
    # Timestep-First Denoising Loop
    # ================================================================

    def _timestep_first_loop(
        self,
        batch_size: int,
        noise: torch.Tensor,
        output: torch.Tensor,
        noisy_inputs: list,
        block_start_frames: list,
        all_num_frames: list,
        exit_flags: list,
        num_denoising_steps: int,
        start_gradient_frame_index: int,
        initial_latent,
        conditional_dict: dict,
    ) -> int:
        """
        Timestep-First denoising loop: outer loop over timesteps, inner loop over frame blocks.

        Core properties:
        1. All blocks exit at the same timestep (uniformly using exit_flags[0])
        2. KV cache is reset and initial_latent is re-cached after each timestep completes
        3. KV cache is updated immediately after each block is denoised (noisy/clean context)

        Returns:
            unified_exit_step: The unified exit step index
        """
        unified_exit_step = exit_flags[0]
        
        for step_index, current_timestep in enumerate(self.denoising_step_list):
            is_exit_step = (step_index == unified_exit_step)
            is_last_step = (step_index == num_denoising_steps - 1)
            
            for block_index, (start_frame, current_num_frames) in enumerate(zip(block_start_frames, all_num_frames)):
                noisy_input = noisy_inputs[block_index]
                
                timestep = torch.ones(
                    [batch_size, current_num_frames],
                    device=noise.device,
                    dtype=torch.int64) * current_timestep

                need_gradient = is_exit_step and (start_frame >= start_gradient_frame_index)
                
                if not is_exit_step:
                    # Non-exit step: do not compute gradients
                    with torch.no_grad():
                        flow_pred, denoised_pred = self.generator(
                            noisy_image_or_video=noisy_input,
                            conditional_dict=conditional_dict,
                            timestep=timestep,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=start_frame * self.frame_seq_length
                        )

                        next_timestep = self.denoising_step_list[step_index + 1]
                        noisy_inputs[block_index] = self._compute_noisy_next(
                            flow_pred, denoised_pred, noisy_input, timestep,
                            next_timestep, batch_size, current_num_frames, noise.device)

                        # Update KV cache
                        if self.always_clean_context:
                            context_timestep = torch.ones_like(timestep) * self.context_noise
                            cache_input_for_kv = denoised_pred
                        else:
                            context_timestep = torch.ones_like(timestep) * next_timestep
                            cache_input_for_kv = noisy_inputs[block_index]
                        self.generator(
                            noisy_image_or_video=cache_input_for_kv,
                            conditional_dict=conditional_dict,
                            timestep=context_timestep,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=start_frame * self.frame_seq_length,
                        )
                else:
                    # Exit step: decide whether to compute gradients based on position
                    if not need_gradient:
                        with torch.no_grad():
                            flow_pred, denoised_pred = self.generator(
                                noisy_image_or_video=noisy_input,
                                conditional_dict=conditional_dict,
                                timestep=timestep,
                                kv_cache=self.kv_cache1,
                                crossattn_cache=self.crossattn_cache,
                                current_start=start_frame * self.frame_seq_length
                            )
                    else:
                        flow_pred, denoised_pred = self.generator(
                            noisy_image_or_video=noisy_input,
                            conditional_dict=conditional_dict,
                            timestep=timestep,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=start_frame * self.frame_seq_length
                        )

                    # Record output
                    output[:, start_frame:start_frame + current_num_frames] = denoised_pred

                    # Update KV cache
                    if is_last_step or self.always_clean_context:
                        context_timestep = torch.ones_like(timestep) * self.context_noise
                        cache_input = denoised_pred
                    else:
                        next_timestep = self.denoising_step_list[step_index + 1]
                        cache_input = self._compute_noisy_next(
                            flow_pred, denoised_pred, noisy_input, timestep,
                            next_timestep, batch_size, current_num_frames, noise.device)
                        context_timestep = torch.ones_like(timestep) * next_timestep
                    
                    with torch.no_grad():
                        self.generator(
                            noisy_image_or_video=cache_input,
                            conditional_dict=conditional_dict,
                            timestep=context_timestep,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=start_frame * self.frame_seq_length,
                        )
            
            if is_exit_step:
                break
            
            # After each timestep completes, reset KV cache so the next timestep can rebuild it
            self._reset_kv_cache(noise.device)
            self._recache_initial_latent(initial_latent, conditional_dict)
        
        return unified_exit_step

    # ================================================================
    # Frame-First Denoising Loop
    # ================================================================

    def _frame_first_loop(
        self,
        batch_size: int,
        noise: torch.Tensor,
        output: torch.Tensor,
        noisy_inputs: list,
        block_start_frames: list,
        all_num_frames: list,
        exit_flags: list,
        num_denoising_steps: int,
        start_gradient_frame_index: int,
        conditional_dict: dict,
    ) -> int:
        """
        Frame-First denoising loop: outer loop over frame blocks, inner loop over timesteps.

        Differences from Timestep-First:
        1. Each block completes all denoising steps before updating KV cache once
        2. KV cache is not written during the inner loop (only written once after the block's exit step)
        3. No need to reset KV cache at each step (KV cache naturally grows between blocks)
        4. Each block can have an independent exit step (exit_flags[block_index])

        Apart from the dimension order, all other logic (noising method, context type, gradient control) is the same as Timestep-First.

        Returns:
            unified_exit_step: Exit step index (takes exit_flags[0], used for subsequent timestep computation)
        """
        for block_index, (start_frame, current_num_frames) in enumerate(zip(block_start_frames, all_num_frames)):
            noisy_input = noisy_inputs[block_index]
            block_exit_step = exit_flags[block_index]
            
            # Used to store the exit step's flow_pred (may be needed for computing noisy context)
            exit_flow_pred = None
            exit_denoised_pred = None
            exit_noisy_input = None
            exit_timestep = None
            
            # Inner loop: iterate over each denoising step
            for step_index, current_timestep in enumerate(self.denoising_step_list):
                is_exit_step = (step_index == block_exit_step)
                is_last_denoising_step = (step_index == num_denoising_steps - 1)
                
                timestep = torch.ones(
                    [batch_size, current_num_frames],
                    device=noise.device,
                    dtype=torch.int64) * current_timestep

                need_gradient = is_exit_step and (start_frame >= start_gradient_frame_index)
                
                if not is_exit_step:
                    # Non-exit step: do not compute gradients, do not update KV cache
                    with torch.no_grad():
                        flow_pred, denoised_pred = self.generator(
                            noisy_image_or_video=noisy_input,
                            conditional_dict=conditional_dict,
                            timestep=timestep,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=start_frame * self.frame_seq_length
                        )

                        # Compute the noisy input for the next step (do not update KV cache)
                        next_timestep = self.denoising_step_list[step_index + 1]
                        noisy_input = self._compute_noisy_next(
                            flow_pred, denoised_pred, noisy_input, timestep,
                            next_timestep, batch_size, current_num_frames, noise.device)
                else:
                    # Exit step: decide whether to compute gradients based on position
                    if not need_gradient:
                        with torch.no_grad():
                            flow_pred, denoised_pred = self.generator(
                                noisy_image_or_video=noisy_input,
                                conditional_dict=conditional_dict,
                                timestep=timestep,
                                kv_cache=self.kv_cache1,
                                crossattn_cache=self.crossattn_cache,
                                current_start=start_frame * self.frame_seq_length
                            )
                    else:
                        flow_pred, denoised_pred = self.generator(
                            noisy_image_or_video=noisy_input,
                            conditional_dict=conditional_dict,
                            timestep=timestep,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=start_frame * self.frame_seq_length
                        )
                    
                    # Record the exit step's information for subsequent KV cache update
                    exit_flow_pred = flow_pred
                    exit_denoised_pred = denoised_pred
                    exit_noisy_input = noisy_input
                    exit_timestep = timestep
                    
                    # Record output
                    output[:, start_frame:start_frame + current_num_frames] = denoised_pred
                    break  # Exit the current block's denoising loop
            
            # After block exits: update KV cache once
            # Logic is exactly the same as the exit step KV cache update in Timestep-First
            if exit_denoised_pred is not None:
                is_last_possible_step = (block_exit_step == num_denoising_steps - 1)
                if is_last_possible_step or self.always_clean_context:
                    context_timestep = torch.ones_like(exit_timestep) * self.context_noise
                    cache_input = exit_denoised_pred
                else:
                    next_timestep = self.denoising_step_list[block_exit_step + 1]
                    cache_input = self._compute_noisy_next(
                        exit_flow_pred, exit_denoised_pred, exit_noisy_input, exit_timestep,
                        next_timestep, batch_size, current_num_frames, noise.device)
                    context_timestep = torch.ones_like(exit_timestep) * next_timestep
                
                with torch.no_grad():
                    self.generator(
                        noisy_image_or_video=cache_input,
                        conditional_dict=conditional_dict,
                        timestep=context_timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=start_frame * self.frame_seq_length,
                    )
        
        # Return exit_flags[0] for subsequent timestep computation (maintaining interface consistency)
        return exit_flags[0]

    def _reset_kv_cache(self, device: torch.device):
        """Reset KV cache."""
        for block_index in range(self.num_transformer_blocks):
            self.crossattn_cache[block_index]["is_init"] = False
        for block_index in range(len(self.kv_cache1)):
            self.kv_cache1[block_index]["global_end_index"] = torch.tensor(
                [0], dtype=torch.long, device=device)
            self.kv_cache1[block_index]["local_end_index"] = torch.tensor(
                [0], dtype=torch.long, device=device)

    def _recache_initial_latent(
            self,
            initial_latent: Optional[torch.Tensor],
            conditional_dict: dict
    ):
        """Re-cache initial_latent (keeping logic consistent with Step 2)."""
        if initial_latent is None:
            return
        
        batch_size = initial_latent.shape[0]
        device = initial_latent.device
        
        # Note: during training, initial_latent shape is [B, 1, C, H, W] (only the first frame)
        # timestep shape needs to match the number of frames in initial_latent
        num_input_frames = initial_latent.shape[1]
        timestep = torch.ones([batch_size, num_input_frames], device=device, dtype=torch.int64) * 0
        
        with torch.no_grad():
            self.generator(
                noisy_image_or_video=initial_latent,
                conditional_dict=conditional_dict,
                timestep=timestep,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=0
            )

    def _initialize_kv_cache(self, batch_size: int, dtype: torch.dtype, device: torch.device):
        """Initialize KV cache for the Wan model."""
        kv_cache1 = []

        for _ in range(self.num_transformer_blocks):
            kv_cache1.append({
                "k": torch.zeros([batch_size, self.kv_cache_size, self.num_attention_heads, self.attention_head_dim], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, self.kv_cache_size, self.num_attention_heads, self.attention_head_dim], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })

        self.kv_cache1 = kv_cache1

    def _initialize_crossattn_cache(self, batch_size: int, dtype: torch.dtype, device: torch.device):
        """Initialize cross-attention cache for the Wan model."""
        crossattn_cache = []

        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append({
                "k": torch.zeros([batch_size, self.text_context_len, self.num_attention_heads, self.attention_head_dim], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, self.text_context_len, self.num_attention_heads, self.attention_head_dim], dtype=dtype, device=device),
                "is_init": False
            })
        self.crossattn_cache = crossattn_cache
