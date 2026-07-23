from typing import Tuple
from einops import rearrange
from torch import nn
import torch.distributed as dist
import torch

from pipeline import SelfForcingTrainingPipeline
from utils.loss import get_denoising_loss
from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper


class BaseModel(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self._initialize_models(args, device)

        self.device = device
        self.args = args
        self.dtype = torch.bfloat16 if args.mixed_precision else torch.float32
        if hasattr(args, "denoising_step_list"):
            self.denoising_step_list = torch.tensor(args.denoising_step_list, dtype=torch.long)
            if args.warp_denoising_step:
                timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
                self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

    def _initialize_models(self, args, device):
        self.real_model_name = getattr(args, "real_name", "Wan2.1-T2V-1.3B")
        self.fake_model_name = getattr(args, "fake_name", "Wan2.1-T2V-1.3B")

        self.generator = WanDiffusionWrapper(**getattr(args, "model_kwargs", {}), is_causal=True)
        self.generator.model.requires_grad_(True)

        self.real_score = WanDiffusionWrapper(model_name=self.real_model_name, is_causal=False)
        self.real_score.model.requires_grad_(False)

        self.fake_score = WanDiffusionWrapper(model_name=self.fake_model_name, is_causal=False)
        self.fake_score.model.requires_grad_(True)

        self.bidirectional_cd_enabled = bool(
            getattr(args, "bidirectional_cd_enabled", False)
        )
        self.bidirectional_cd_ema = None
        if self.bidirectional_cd_enabled:
            # A separate frozen target is required: swapping EMA weights into
            # the live student between forward and backward invalidates
            # autograd/FSDP version counters.
            self.bidirectional_cd_ema = WanDiffusionWrapper(
                model_name=self.fake_model_name,
                is_causal=True,
                empty_init=True,
                **getattr(args, "model_kwargs", {}),
            )
            self.bidirectional_cd_ema.requires_grad_(False)
            self.bidirectional_cd_ema.eval()

        self.text_encoder = WanTextEncoder()
        self.text_encoder.requires_grad_(False)

        self.vae = WanVAEWrapper()
        self.vae.requires_grad_(False)

        self.scheduler = self.generator.get_scheduler()
        self.scheduler.timesteps = self.scheduler.timesteps.to(device)

    def _get_timestep(
            self,
            min_timestep: int,
            max_timestep: int,
            batch_size: int,
            num_frame: int,
            num_frame_per_block: int,
            uniform_timestep: bool = False
    ) -> torch.Tensor:
        """
        Randomly generate a timestep tensor based on the generator's task type. It uniformly samples a timestep
        from the range [min_timestep, max_timestep], and returns a tensor of shape [batch_size, num_frame].
        - If uniform_timestep, it will use the same timestep for all frames.
        - If not uniform_timestep, it will use a different timestep for each block.
        """
        if uniform_timestep:
            timestep = torch.randint(
                min_timestep,
                max_timestep,
                [batch_size, 1],
                device=self.device,
                dtype=torch.long
            ).repeat(1, num_frame)
            return timestep
        else:
            timestep = torch.randint(
                min_timestep,
                max_timestep,
                [batch_size, num_frame],
                device=self.device,
                dtype=torch.long
            )
            # make the noise level the same within every block
            if self.independent_first_frame:
                # the first frame is always kept the same
                timestep_from_second = timestep[:, 1:]
                timestep_from_second = timestep_from_second.reshape(
                    timestep_from_second.shape[0], -1, num_frame_per_block)
                timestep_from_second[:, :, 1:] = timestep_from_second[:, :, 0:1]
                timestep_from_second = timestep_from_second.reshape(
                    timestep_from_second.shape[0], -1)
                timestep = torch.cat([timestep[:, 0:1], timestep_from_second], dim=1)
            else:
                timestep = timestep.reshape(
                    timestep.shape[0], -1, num_frame_per_block)
                timestep[:, :, 1:] = timestep[:, :, 0:1]
                timestep = timestep.reshape(timestep.shape[0], -1)
            return timestep


class SelfForcingModel(BaseModel):
    def __init__(self, args, device):
        super().__init__(args, device)
        self.denoising_loss_func = get_denoising_loss(args.denoising_loss_type)()
        # Default values, subclasses (DMD/TimestepForcingDMD) will override
        if not hasattr(self, 'num_training_frames'):
            self.num_training_frames = getattr(args, "num_training_frames", 21)
        if not hasattr(self, 'num_frame_per_block'):
            self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        if not hasattr(self, 'num_rollout_frames'):
            _rollout = getattr(args, "num_rollout_frames", 0)
            self.num_rollout_frames = _rollout if _rollout > 0 else self.num_training_frames

    def _run_generator(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        initial_latent: torch.tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Optionally simulate the generator's input from noise using backward simulation
        and then run the generator for one-step.
        Input:
            - image_or_video_shape: a list containing the shape of the image or video [B, F, C, H, W].
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - clean_latent: a tensor containing the clean latents [B, F, C, H, W]. Need to be passed when no backward simulation is used.
            - initial_latent: a tensor containing the initial latents [B, F, C, H, W].
        Output:
            - pred_image: a tensor with shape [B, F, C, H, W].
            - denoised_timestep: an integer
        """
        # Step 1: Sample noise and backward simulate the generator's input
        assert getattr(self.args, "backward_simulation", True), "Backward simulation needs to be enabled"
        conditional_dict = dict(conditional_dict)
        if initial_latent is not None:
            conditional_dict["initial_latent"] = initial_latent

        # During training, the number of generated frames should be uniformly sampled from
        # [num_training_frames, num_rollout_frames], but still being a multiple of num_frame_per_block
        num_grad_frames = self.num_training_frames  # Number of frames for gradient/loss computation (e.g., 21)
        context_frames = 1 if initial_latent is not None else 0
        layout_offset = int(
            context_frames > 0 or bool(self.args.independent_first_frame)
        )
        min_block_frames = num_grad_frames - layout_offset
        max_block_frames = self.num_rollout_frames - layout_offset
        if (
            min_block_frames <= 0
            or min_block_frames % self.num_frame_per_block
            or max_block_frames % self.num_frame_per_block
        ):
            raise ValueError(
                "Invalid rollout layout: total frame counts minus the "
                f"{layout_offset}-frame context/singleton prefix must be "
                f"multiples of block size {self.num_frame_per_block}"
            )
        max_num_blocks = max_block_frames // self.num_frame_per_block
        min_num_blocks = min_block_frames // self.num_frame_per_block
        fixed_rollout_frames = int(
            getattr(self.args, "fixed_rollout_frames", 0)
        )
        if fixed_rollout_frames < 0:
            raise ValueError("fixed_rollout_frames must be non-negative")
        if fixed_rollout_frames > 0:
            fixed_block_frames = fixed_rollout_frames - layout_offset
            if (
                fixed_rollout_frames < num_grad_frames
                or fixed_rollout_frames > self.num_rollout_frames
                or fixed_block_frames % self.num_frame_per_block
            ):
                raise ValueError(
                    "fixed_rollout_frames must be within "
                    f"[{num_grad_frames}, {self.num_rollout_frames}] and its "
                    f"layout-adjusted length must be divisible by block size "
                    f"{self.num_frame_per_block}; got {fixed_rollout_frames}"
                )
            num_generated_blocks = torch.tensor(
                [fixed_block_frames // self.num_frame_per_block],
                device=self.device,
                dtype=torch.long,
            )
        else:
            num_generated_blocks = torch.randint(
                min_num_blocks,
                max_num_blocks + 1,
                (1,),
                device=self.device,
            )
        dist.broadcast(num_generated_blocks, src=0)
        num_generated_blocks = num_generated_blocks.item()
        num_output_frames = (
            layout_offset + num_generated_blocks * self.num_frame_per_block
        )
        # I2V supplies its context separately; independent-first-frame T2V
        # includes the singleton first frame in the sampled noise itself.
        num_noise_frames = num_output_frames - context_frames
        noise_shape = [
            image_or_video_shape[0],
            num_noise_frames,
            *image_or_video_shape[2:],
        ]

        pred_image_or_video, denoised_timestep_from, denoised_timestep_to = self._consistency_backward_simulation(
            noise=torch.randn(noise_shape,
                              device=self.device, dtype=self.dtype),
            **conditional_dict,
        )
        # Slice last num_training_frames frames for loss computation
        num_tail = num_grad_frames - 1  # Number of tail frames when concatenating using VAE re-encoding
        did_reencode = pred_image_or_video.shape[1] > num_grad_frames
        if did_reencode:
            with torch.no_grad():
                # Reencode to get image latent at the boundary
                latent_to_decode = pred_image_or_video[:, :-num_tail, ...]
                # Decode to video
                frame = self.vae.decode_last_frame(latent_to_decode).to(self.dtype)
                frame = rearrange(frame, "b t c h w -> b c t h w")
                # Encode frame to get image latent
                image_latent = self.vae.encode_to_latent(frame).to(self.dtype)
            pred_image_or_video_last_21 = torch.cat([image_latent, pred_image_or_video[:, -num_tail:, ...]], dim=1)
        else:
            pred_image_or_video_last_21 = pred_image_or_video

        if did_reencode or initial_latent is not None:
            # The re-encoded boundary (or external I2V context) is detached.
            # In the regular block layout the remainder of its partial block is
            # also excluded so every active region is a complete causal block.
            gradient_mask = torch.ones_like(pred_image_or_video_last_21, dtype=torch.bool)
            if did_reencode:
                masked_prefix = (
                    1
                    if self.args.independent_first_frame or initial_latent is not None
                    else self.num_frame_per_block
                )
            else:
                masked_prefix = initial_latent.shape[1]
            gradient_mask[:, :masked_prefix] = False
        else:
            gradient_mask = None

        if gradient_mask is None:
            active_gradient_frames = num_grad_frames
        else:
            # The mask is spatially/channel uniform by construction; record
            # the per-sample active temporal support for auditable smokes.
            active_gradient_frames = int(
                gradient_mask[:, :, 0, 0, 0]
                .sum(dim=1)
                .float()
                .mean()
                .item()
            )
        self._last_rollout_metadata = {
            "rollout_actual_frames": torch.tensor(
                float(num_output_frames), device=self.device
            ),
            "rollout_did_reencode": torch.tensor(
                float(did_reencode), device=self.device
            ),
            "rollout_active_gradient_frames": torch.tensor(
                float(active_gradient_frames), device=self.device
            ),
        }

        pred_image_or_video_last_21 = pred_image_or_video_last_21.to(self.dtype)
        return pred_image_or_video_last_21, gradient_mask, denoised_timestep_from, denoised_timestep_to

    def _consistency_backward_simulation(
        self,
        noise: torch.Tensor,
        **conditional_dict: dict
    ) -> torch.Tensor:
        """
        Simulate the generator's input from noise to avoid training/inference mismatch.
        See Sec 4.5 of the DMD2 paper (https://arxiv.org/abs/2405.14867) for details.
        Here we use the consistency sampler (https://arxiv.org/abs/2303.01469)
        Input:
            - noise: a tensor sampled from N(0, 1) with shape [B, F, C, H, W] where the number of frame is 1 for images.
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
        Output:
            - output: a tensor with shape [B, T, F, C, H, W].
            T is the total number of timesteps. output[0] is a pure noise and output[i] and i>0
            represents the x0 prediction at each timestep.
        """
        if self.inference_pipeline is None:
            self._initialize_inference_pipeline()

        return self.inference_pipeline.inference_with_trajectory(
            noise=noise, **conditional_dict
        )

    def _initialize_inference_pipeline(self):
        """
        Lazy initialize the inference pipeline during the first backward simulation run.
        Here we encapsulate the inference code with a model-dependent outside function.
        We pass our FSDP-wrapped modules into the pipeline to save memory.
        """
        self.inference_pipeline = SelfForcingTrainingPipeline(
            denoising_step_list=self.denoising_step_list,
            scheduler=self.scheduler,
            generator=self.generator,
            num_frame_per_block=self.num_frame_per_block,
            independent_first_frame=self.args.independent_first_frame,
            same_step_across_blocks=self.args.same_step_across_blocks,
            last_step_only=self.args.last_step_only,
            num_max_frames=self.num_rollout_frames,
            num_gradient_frames=self.num_training_frames,
            context_noise=self.args.context_noise
        )
