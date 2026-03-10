from typing import List, Optional
import torch
from tqdm import tqdm
from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper

from utils.memory import get_cuda_free_memory_gb, DynamicSwapInstaller, move_model_to_device_with_memory_preservation


class CausalInferencePipeline(torch.nn.Module):
    def __init__(
            self,
            args,
            device,
            generator=None,
            text_encoder=None,
            vae=None
    ):
        super().__init__()
        # Step 1: Initialize all models
        self.generator = WanDiffusionWrapper(
            **getattr(args, "model_kwargs", {}), is_causal=True) if generator is None else generator
        self.text_encoder = WanTextEncoder() if text_encoder is None else text_encoder
        self.vae = WanVAEWrapper() if vae is None else vae

        # Step 2: Initialize all causal hyperparmeters
        self.scheduler = self.generator.get_scheduler()
        self.denoising_step_list = torch.tensor(
            args.denoising_step_list, dtype=torch.long)
        if args.warp_denoising_step:
            timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
            self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

        model = self.generator.model
        self.num_transformer_blocks = len(model.blocks)
        self.num_attention_heads = model.num_heads
        self.attention_head_dim = model.dim // model.num_heads
        self.text_context_len = model.text_len
        self.patch_size = model.patch_size
        self.frame_seq_length = None

        self.kv_cache1 = None
        self.args = args
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.independent_first_frame = args.independent_first_frame
        self.local_attn_size = model.local_attn_size
        self.use_ode_trajectory = getattr(args, "use_ode_trajectory", False)
        self.always_clean_context = getattr(args, "always_clean_context", False)

        print(f"KV inference with {self.num_frame_per_block} frames per block")
        if self.use_ode_trajectory:
            print("Using ODE trajectory (deterministic) for inference")

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

    def _update_runtime_cache_spec(self, height, width):
        patch_h = self.patch_size[1]
        patch_w = self.patch_size[2]
        if height % patch_h != 0 or width % patch_w != 0:
            raise ValueError(
                f"Latent spatial size ({height}, {width}) is incompatible with patch size {self.patch_size}"
            )
        self.frame_seq_length = (height // patch_h) * (width // patch_w)

    def inference(
        self,
        noise: torch.Tensor,
        text_prompts: List[str],
        initial_latent: Optional[torch.Tensor] = None,
        return_latents: bool = False,
        profile: bool = False,
        low_memory: bool = False,
    ) -> torch.Tensor:
        """
        Perform inference on the given noise and text prompts.
        Inputs:
            noise (torch.Tensor): The input noise tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
            text_prompts (List[str]): The list of text prompts.
            initial_latent (torch.Tensor): The initial latent tensor of shape
                (batch_size, num_input_frames, num_channels, height, width).
                If num_input_frames is 1, perform image to video.
                If num_input_frames is greater than 1, perform video extension.
            return_latents (bool): Whether to return the latents.
        Outputs:
            video (torch.Tensor): The generated video tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
                It is normalized to be in the range [0, 1].
        """
        batch_size, num_frames, num_channels, height, width = noise.shape
        self._update_runtime_cache_spec(height, width)
        if not self.independent_first_frame or (self.independent_first_frame and initial_latent is not None):
            # If the first frame is independent and the first frame is provided, then the number of frames in the
            # noise should still be a multiple of num_frame_per_block
            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block
        else:
            # Using a [1, 4, 4, 4, 4, 4, ...] model to generate a video without image conditioning
            assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block
        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames  # add the initial latent frames
        conditional_dict = self.text_encoder(
            text_prompts=text_prompts
        )

        if low_memory:
            gpu_memory_preservation = get_cuda_free_memory_gb(noise.device) + 5
            move_model_to_device_with_memory_preservation(self.text_encoder, target_device=noise.device, preserved_memory_gb=gpu_memory_preservation)

        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype
        )

        # Set up profiling if requested
        if profile:
            init_start = torch.cuda.Event(enable_timing=True)
            init_end = torch.cuda.Event(enable_timing=True)
            diffusion_start = torch.cuda.Event(enable_timing=True)
            diffusion_end = torch.cuda.Event(enable_timing=True)
            vae_start = torch.cuda.Event(enable_timing=True)
            vae_end = torch.cuda.Event(enable_timing=True)
            block_times = []
            block_start = torch.cuda.Event(enable_timing=True)
            block_end = torch.cuda.Event(enable_timing=True)
            init_start.record()

        # Step 1: Initialize KV cache to all zeros
        if self.kv_cache1 is None:
            self._initialize_kv_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )
            self._initialize_crossattn_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )
        else:
            # reset cross attn cache
            for block_index in range(self.num_transformer_blocks):
                self.crossattn_cache[block_index]["is_init"] = False
            # reset kv cache
            for block_index in range(len(self.kv_cache1)):
                self.kv_cache1[block_index]["global_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                self.kv_cache1[block_index]["local_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)

        # Step 2: Cache context feature
        current_start_frame = 0
        if initial_latent is not None:
            timestep = torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0
            if self.independent_first_frame:
                # Assume num_input_frames is 1 + self.num_frame_per_block * num_input_blocks
                assert (num_input_frames - 1) % self.num_frame_per_block == 0
                num_input_blocks = (num_input_frames - 1) // self.num_frame_per_block
                output[:, :1] = initial_latent[:, :1]
                self.generator(
                    noisy_image_or_video=initial_latent[:, :1],
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                )
                current_start_frame += 1
            else:
                # Assume num_input_frames is self.num_frame_per_block * num_input_blocks
                assert num_input_frames % self.num_frame_per_block == 0
                num_input_blocks = num_input_frames // self.num_frame_per_block

            for _ in range(num_input_blocks):
                current_ref_latents = \
                    initial_latent[:, current_start_frame:current_start_frame + self.num_frame_per_block]
                output[:, current_start_frame:current_start_frame + self.num_frame_per_block] = current_ref_latents
                self.generator(
                    noisy_image_or_video=current_ref_latents,
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                )
                current_start_frame += self.num_frame_per_block

        if profile:
            init_end.record()
            torch.cuda.synchronize()
            diffusion_start.record()

        # Step 3: Temporal denoising loop
        all_num_frames = [self.num_frame_per_block] * num_blocks
        if self.independent_first_frame and initial_latent is None:
            all_num_frames = [1] + all_num_frames
        for current_num_frames in all_num_frames:
            if profile:
                block_start.record()

            noisy_input = noise[
                :, current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames]

            # Step 3.1: Spatial denoising loop
            for index, current_timestep in enumerate(self.denoising_step_list):
                # print(f"current_timestep: {current_timestep}")
                # set current timestep
                timestep = torch.ones(
                    [batch_size, current_num_frames],
                    device=noise.device,
                    dtype=torch.int64) * current_timestep

                if index < len(self.denoising_step_list) - 1:
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length
                    )
                    next_timestep = self.denoising_step_list[index + 1]
                    noisy_input = self.scheduler.add_noise(
                        denoised_pred.flatten(0, 1),
                        torch.randn_like(denoised_pred.flatten(0, 1)),
                        next_timestep * torch.ones(
                            [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
                    ).unflatten(0, denoised_pred.shape[:2])
                else:
                    # for getting real output
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length
                    )

            # Step 3.2: record the model's output
            output[:, current_start_frame:current_start_frame + current_num_frames] = denoised_pred

            # Step 3.3: rerun with timestep zero to update KV cache using clean context
            context_timestep = torch.ones_like(timestep) * self.args.context_noise
            self.generator(
                noisy_image_or_video=denoised_pred,
                conditional_dict=conditional_dict,
                timestep=context_timestep,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=current_start_frame * self.frame_seq_length,
            )

            if profile:
                block_end.record()
                torch.cuda.synchronize()
                block_time = block_start.elapsed_time(block_end)
                block_times.append(block_time)

            # Step 3.4: update the start and end frame indices
            current_start_frame += current_num_frames

        if profile:
            # End diffusion timing and synchronize CUDA
            diffusion_end.record()
            torch.cuda.synchronize()
            diffusion_time = diffusion_start.elapsed_time(diffusion_end)
            init_time = init_start.elapsed_time(init_end)
            vae_start.record()

        # Step 4: Decode the output
        video = self.vae.decode_to_pixel(output, use_cache=False)
        video = (video * 0.5 + 0.5).clamp(0, 1)

        if profile:
            # End VAE timing and synchronize CUDA
            vae_end.record()
            torch.cuda.synchronize()
            vae_time = vae_start.elapsed_time(vae_end)
            total_time = init_time + diffusion_time + vae_time

            print("Profiling results:")
            print(f"  - Initialization/caching time: {init_time:.2f} ms ({100 * init_time / total_time:.2f}%)")
            print(f"  - Diffusion generation time: {diffusion_time:.2f} ms ({100 * diffusion_time / total_time:.2f}%)")
            for i, block_time in enumerate(block_times):
                print(f"    - Block {i} generation time: {block_time:.2f} ms ({100 * block_time / diffusion_time:.2f}% of diffusion)")
            print(f"  - VAE decoding time: {vae_time:.2f} ms ({100 * vae_time / total_time:.2f}%)")
            print(f"  - Total time: {total_time:.2f} ms")

        if return_latents:
            return video, output
        else:
            return video

    def _initialize_kv_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        kv_cache1 = []
        if self.local_attn_size != -1:
            # Use the local attention size to compute the KV cache size
            kv_cache_size = self.local_attn_size * self.frame_seq_length
        else:
            # Use the default KV cache size
            kv_cache_size = 32760

        for _ in range(self.num_transformer_blocks):
            kv_cache1.append({
                "k": torch.zeros([batch_size, kv_cache_size, self.num_attention_heads, self.attention_head_dim], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, kv_cache_size, self.num_attention_heads, self.attention_head_dim], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })

        self.kv_cache1 = kv_cache1  # always store the clean cache

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """
        crossattn_cache = []

        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append({
                "k": torch.zeros([batch_size, self.text_context_len, self.num_attention_heads, self.attention_head_dim], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, self.text_context_len, self.num_attention_heads, self.attention_head_dim], dtype=dtype, device=device),
                "is_init": False
            })
        self.crossattn_cache = crossattn_cache

    def inference_hybrid(
        self,
        noise: torch.Tensor,
        text_prompts: List[str],
        initial_latent: Optional[torch.Tensor] = None,
        return_latents: bool = False,
        profile: bool = False,
        low_memory: bool = False,
        frame_first_steps: int = 0,
        early_return_step: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Perform inference with customizable denoising order.
        
        This method supports hybrid denoising strategies:
        - First `frame_first_steps` steps: use frame-first order (frame outer loop, timestep inner loop)
        - Remaining steps: use timestep-first order (timestep outer loop, frame inner loop)
        
        Additionally, supports early return at a specific step.
        
        Inputs:
            noise (torch.Tensor): The input noise tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
            text_prompts (List[str]): The list of text prompts.
            initial_latent (torch.Tensor): The initial latent tensor of shape
                (batch_size, num_input_frames, num_channels, height, width).
                If num_input_frames is 1, perform image to video.
                If num_input_frames is greater than 1, perform video extension.
            return_latents (bool): Whether to return the latents.
            profile (bool): Whether to profile the inference time.
            low_memory (bool): Whether to use low memory mode.
            frame_first_steps (int): Number of initial steps to use frame-first denoising order.
                Default is 0 (all steps use timestep-first order).
                If set to total_steps, all steps use frame-first order (same as original inference).
            early_return_step (int): If specified, return after completing this step (0-indexed).
                For example, if total_steps=4 and early_return_step=2, 
                only steps 0, 1, 2 will be executed and step 3 will be skipped.
                If None, all steps will be executed.
        Outputs:
            video (torch.Tensor): The generated video tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
                It is normalized to be in the range [0, 1].
        """
        batch_size, num_frames, num_channels, height, width = noise.shape
        self._update_runtime_cache_spec(height, width)
        if not self.independent_first_frame or (self.independent_first_frame and initial_latent is not None):
            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block
        else:
            assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block
        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames
        num_denoising_steps = len(self.denoising_step_list)
        
        # Validate parameters
        if early_return_step is not None:
            assert 0 <= early_return_step < num_denoising_steps, \
                f"early_return_step must be in [0, {num_denoising_steps - 1}], got {early_return_step}"
        assert 0 <= frame_first_steps <= num_denoising_steps, \
            f"frame_first_steps must be in [0, {num_denoising_steps}], got {frame_first_steps}"
        
        # Compute the actual number of steps to execute
        actual_last_step = early_return_step if early_return_step is not None else (num_denoising_steps - 1)
        
        conditional_dict = self.text_encoder(
            text_prompts=text_prompts
        )

        if low_memory:
            gpu_memory_preservation = get_cuda_free_memory_gb(noise.device) + 5
            move_model_to_device_with_memory_preservation(self.text_encoder, target_device=noise.device, preserved_memory_gb=gpu_memory_preservation)

        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype
        )

        # Set up profiling if requested
        if profile:
            init_start = torch.cuda.Event(enable_timing=True)
            init_end = torch.cuda.Event(enable_timing=True)
            diffusion_start = torch.cuda.Event(enable_timing=True)
            diffusion_end = torch.cuda.Event(enable_timing=True)
            vae_start = torch.cuda.Event(enable_timing=True)
            vae_end = torch.cuda.Event(enable_timing=True)
            step_times = []
            step_start = torch.cuda.Event(enable_timing=True)
            step_end = torch.cuda.Event(enable_timing=True)
            init_start.record()

        # Step 1: Initialize KV cache to all zeros
        if self.kv_cache1 is None:
            self._initialize_kv_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )
            self._initialize_crossattn_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )
        else:
            for block_index in range(self.num_transformer_blocks):
                self.crossattn_cache[block_index]["is_init"] = False
            for block_index in range(len(self.kv_cache1)):
                self.kv_cache1[block_index]["global_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                self.kv_cache1[block_index]["local_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)

        # Step 2: Cache context feature (same as original)
        current_start_frame = 0
        num_input_blocks = 0
        if initial_latent is not None:
            timestep = torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0
            if self.independent_first_frame:
                assert (num_input_frames - 1) % self.num_frame_per_block == 0
                num_input_blocks = (num_input_frames - 1) // self.num_frame_per_block
                output[:, :1] = initial_latent[:, :1]
                self.generator(
                    noisy_image_or_video=initial_latent[:, :1],
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                )
                current_start_frame += 1
            else:
                assert num_input_frames % self.num_frame_per_block == 0
                num_input_blocks = num_input_frames // self.num_frame_per_block

            for _ in range(num_input_blocks):
                current_ref_latents = \
                    initial_latent[:, current_start_frame:current_start_frame + self.num_frame_per_block]
                output[:, current_start_frame:current_start_frame + self.num_frame_per_block] = current_ref_latents
                self.generator(
                    noisy_image_or_video=current_ref_latents,
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                )
                current_start_frame += self.num_frame_per_block

        if profile:
            init_end.record()
            torch.cuda.synchronize()
            diffusion_start.record()

        # Step 3: Prepare frame block info
        all_num_frames = [self.num_frame_per_block] * num_blocks
        if self.independent_first_frame and initial_latent is None:
            all_num_frames = [1] + all_num_frames
        
        # Compute the start frame index for each block
        block_start_frames = []
        temp_start = current_start_frame
        for nf in all_num_frames:
            block_start_frames.append(temp_start)
            temp_start += nf

        # Initialize noisy input and denoised prediction for each block
        # noisy_inputs[i] stores the current noisy latent for the i-th block
        # denoised_preds[i] stores the current denoised prediction for the i-th block
        noisy_inputs = []
        denoised_preds = [None] * len(all_num_frames)
        for i, (start_frame, nf) in enumerate(zip(block_start_frames, all_num_frames)):
            noisy_input = noise[:, start_frame - num_input_frames:start_frame + nf - num_input_frames]
            noisy_inputs.append(noisy_input)

        # Helper function: reset KV cache
        def reset_kv_cache():
            for block_idx in range(self.num_transformer_blocks):
                self.crossattn_cache[block_idx]["is_init"] = False
            for block_idx in range(len(self.kv_cache1)):
                self.kv_cache1[block_idx]["global_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                self.kv_cache1[block_idx]["local_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)

        # Helper function: re-cache initial_latent
        def recache_initial_latent():
            if initial_latent is not None:
                temp_start_frame = 0
                init_timestep = torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0
                if self.independent_first_frame:
                    self.generator(
                        noisy_image_or_video=initial_latent[:, :1],
                        conditional_dict=conditional_dict,
                        timestep=init_timestep * 0,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=temp_start_frame * self.frame_seq_length,
                    )
                    temp_start_frame += 1
                
                for _ in range(num_input_blocks):
                    current_ref_latents = \
                        initial_latent[:, temp_start_frame:temp_start_frame + self.num_frame_per_block]
                    self.generator(
                        noisy_image_or_video=current_ref_latents,
                        conditional_dict=conditional_dict,
                        timestep=init_timestep * 0,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=temp_start_frame * self.frame_seq_length,
                    )
                    temp_start_frame += self.num_frame_per_block

        # Step 4: Hybrid denoising loop
        # First frame_first_steps steps use frame-first order
        # Remaining steps use timestep-first order
        
        current_step = 0
        
        # ========== Phase 1: Frame-first denoising (first k steps) ==========
        if frame_first_steps > 0:
            # Iterate over frame blocks in order
            for block_index, (start_frame, current_num_frames) in tqdm(enumerate(zip(block_start_frames, all_num_frames))):
                if profile:
                    step_start.record()
                
                noisy_input = noisy_inputs[block_index]

                cur_actual_last_step = 0
                
                # Denoise the current block for frame_first_steps steps
                for step_index in range(frame_first_steps):
                    cur_actual_last_step = step_index
                    current_timestep = self.denoising_step_list[step_index]
                    is_early_return = (step_index == actual_last_step)
                    is_phase1_last_step = (step_index == frame_first_steps - 1)
                    
                    # Set current timestep
                    timestep = torch.ones(
                        [batch_size, current_num_frames],
                        device=noise.device,
                        dtype=torch.int64) * current_timestep

                    # Perform denoising
                    flow_pred, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=start_frame * self.frame_seq_length
                    )
                    
                    if is_early_return:
                        # Early return: record the output of the current block
                        output[:, start_frame:start_frame + current_num_frames] = denoised_pred
                        denoised_preds[block_index] = denoised_pred
                        real_step_index = min(step_index + 1, num_denoising_steps - 1)
                        next_timestep = self.denoising_step_list[real_step_index]
                        if self.use_ode_trajectory:
                            # ODE trajectory: use scheduler.step (deterministic), pass target_timestep for large step jumps
                            noisy_inputs[block_index] = self.scheduler.step(
                                flow_pred.flatten(0, 1),
                                timestep.flatten(0, 1),
                                noisy_input.flatten(0, 1),
                                target_timestep=next_timestep
                            ).unflatten(0, denoised_pred.shape[:2])
                        else:
                            # SDE trajectory: use add_noise + fresh random noise
                            noisy_inputs[block_index] = self.scheduler.add_noise(
                                denoised_pred.flatten(0, 1),
                                torch.randn_like(denoised_pred.flatten(0, 1)),
                                next_timestep * torch.ones(
                                    [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
                            ).unflatten(0, denoised_pred.shape[:2])
                        break
                    elif is_phase1_last_step:
                        # Phase 1 last step: save denoised_pred and add noise to the next timestep
                        denoised_preds[block_index] = denoised_pred
                        if frame_first_steps < num_denoising_steps and actual_last_step >= frame_first_steps:
                            next_timestep = self.denoising_step_list[frame_first_steps]
                            if self.use_ode_trajectory:
                                # ODE trajectory: use scheduler.step (deterministic), pass target_timestep for large step jumps
                                noisy_inputs[block_index] = self.scheduler.step(
                                    flow_pred.flatten(0, 1),
                                    timestep.flatten(0, 1),
                                    noisy_input.flatten(0, 1),
                                    target_timestep=next_timestep
                                ).unflatten(0, denoised_pred.shape[:2])
                            else:
                                # SDE trajectory: use add_noise + fresh random noise
                                noisy_inputs[block_index] = self.scheduler.add_noise(
                                    denoised_pred.flatten(0, 1),
                                    torch.randn_like(denoised_pred.flatten(0, 1)),
                                    next_timestep * torch.ones(
                                        [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
                                ).unflatten(0, denoised_pred.shape[:2])
                    else:
                        # Add noise to the next timestep
                        next_timestep = self.denoising_step_list[step_index + 1]
                        if self.use_ode_trajectory:
                            # ODE trajectory: use scheduler.step (deterministic), pass target_timestep for large step jumps
                            noisy_input = self.scheduler.step(
                                flow_pred.flatten(0, 1),
                                timestep.flatten(0, 1),
                                noisy_input.flatten(0, 1),
                                target_timestep=next_timestep
                            ).unflatten(0, denoised_pred.shape[:2])
                        else:
                            # SDE trajectory: use add_noise + fresh random noise
                            noisy_input = self.scheduler.add_noise(
                                denoised_pred.flatten(0, 1),
                                torch.randn_like(denoised_pred.flatten(0, 1)),
                                next_timestep * torch.ones(
                                    [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
                            ).unflatten(0, denoised_pred.shape[:2])

                    

                # Update KV cache:
                # - If phase 1 has completed all denoising steps (frame_first_steps == num_denoising_steps), use clean denoised_pred
                # - Otherwise, use the noisy_input with noise added for the next step (maintaining intermediate-step noise level)
                is_fully_denoised = (cur_actual_last_step == num_denoising_steps - 1) or self.always_clean_context
                if is_fully_denoised:
                    # Fully denoised, update KV cache with clean context
                    context_timestep = torch.ones(
                        [batch_size, current_num_frames],
                        device=noise.device,
                        dtype=torch.int64) * self.args.context_noise
                    cache_input = denoised_preds[block_index]
                else:
                    # Not all denoising steps completed, update KV cache with the noised noisy_input
                    # noisy_inputs[block_index] has already been noised to next_timestep above
                    next_timestep = self.denoising_step_list[cur_actual_last_step + 1]
                    context_timestep = torch.ones(
                        [batch_size, current_num_frames],
                        device=noise.device,
                        dtype=torch.int64) * next_timestep
                    cache_input = noisy_inputs[block_index]
                
                self.generator(
                    noisy_image_or_video=cache_input,
                    conditional_dict=conditional_dict,
                    timestep=context_timestep,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=start_frame * self.frame_seq_length,
                )

                if profile:
                    step_end.record()
                    torch.cuda.synchronize()
                    step_time = step_start.elapsed_time(step_end)
                    step_times.append(("frame_first", block_index, step_time))
        
        # Check if early return is needed (already completed in phase 1)
        if early_return_step is not None and early_return_step < frame_first_steps:
            # All blocks have already been processed for early return in phase 1
            pass
        else:
            # ========== Phase 2: Timestep-first denoising (remaining steps) ==========
            if frame_first_steps < num_denoising_steps and actual_last_step >= frame_first_steps:
                # Reset KV cache so timestep-first mode can rebuild it
                reset_kv_cache()
                recache_initial_latent()
                
                for step_index in range(frame_first_steps, actual_last_step + 1):
                    if profile:
                        step_start.record()
                    
                    current_timestep = self.denoising_step_list[step_index]
                    is_last_step = (step_index == actual_last_step)

                    # Inner loop: iterate over each frame block
                    for block_index, (start_frame, current_num_frames) in tqdm(enumerate(zip(block_start_frames, all_num_frames))):
                        noisy_input = noisy_inputs[block_index]

                        # Set current timestep
                        timestep = torch.ones(
                            [batch_size, current_num_frames],
                            device=noise.device,
                            dtype=torch.int64) * current_timestep

                        # Perform denoising
                        flow_pred, denoised_pred = self.generator(
                            noisy_image_or_video=noisy_input,
                            conditional_dict=conditional_dict,
                            timestep=timestep,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=start_frame * self.frame_seq_length
                        )
                        
                        if not is_last_step:
                            # Add noise to the next timestep
                            next_timestep = self.denoising_step_list[step_index + 1]
                            if self.use_ode_trajectory:
                                # ODE trajectory: use scheduler.step (deterministic), pass target_timestep for large step jumps
                                noisy_inputs[block_index] = self.scheduler.step(
                                    flow_pred.flatten(0, 1),
                                    timestep.flatten(0, 1),
                                    noisy_input.flatten(0, 1),
                                    target_timestep=next_timestep
                                ).unflatten(0, denoised_pred.shape[:2])
                            else:
                                # SDE trajectory: use add_noise + fresh random noise
                                noisy_inputs[block_index] = self.scheduler.add_noise(
                                    denoised_pred.flatten(0, 1),
                                    torch.randn_like(denoised_pred.flatten(0, 1)),
                                    next_timestep * torch.ones(
                                        [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
                                ).unflatten(0, denoised_pred.shape[:2])
                            
                            # Update KV cache
                            if self.always_clean_context:
                                # Force using clean context
                                context_timestep = torch.ones_like(timestep) * self.args.context_noise
                                cache_input_for_kv = denoised_pred
                            else:
                                # Use the next step's noisy_input (maintaining intermediate-step noise level)
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
                            # # Last step, record output
                            # output[:, start_frame:start_frame + current_num_frames] = denoised_pred
                            
                            # # Update KV cache: use clean context (fully denoised)
                            # context_timestep = torch.ones_like(timestep) * self.args.context_noise
                            # self.generator(
                            #     noisy_image_or_video=denoised_pred,
                            #     conditional_dict=conditional_dict,
                            #     timestep=context_timestep,
                            #     kv_cache=self.kv_cache1,
                            #     crossattn_cache=self.crossattn_cache,
                            #     current_start=start_frame * self.frame_seq_length,
                            # )

                            
                            # 1. In any case, the output stores the current denoised result
                            output[:, start_frame:start_frame + current_num_frames] = denoised_pred
                            
                            # 2. [Key modification] Decide what to store in KV Cache
                            # Determine if this is physically the last step of the scheduler
                            is_truly_fully_denoised = (step_index == num_denoising_steps - 1) or self.always_clean_context
                            
                            if is_truly_fully_denoised:
                                # Case A: Truly completed all steps -> Cache Clean Latent
                                context_timestep = torch.ones_like(timestep) * self.args.context_noise
                                cache_input = denoised_pred
                            else:
                                # Case B: Early Return, there should be more steps ahead -> Cache Noisy Latent (simulate next step's noise)
                                # This is to stay consistent with Phase 1 logic, preventing subsequent blocks from seeing overly clean context
                                
                                # Get the next timestep (t+1)
                                next_timestep = self.denoising_step_list[step_index + 1]
                                
                                # Manually add noise / ODE step (consistent with Phase 1 logic)
                                if self.use_ode_trajectory:
                                    # ODE trajectory: use scheduler.step (deterministic), pass target_timestep for large step jumps
                                    noisy_next = self.scheduler.step(
                                        flow_pred.flatten(0, 1),
                                        timestep.flatten(0, 1),
                                        noisy_input.flatten(0, 1),
                                        target_timestep=next_timestep
                                    ).unflatten(0, denoised_pred.shape[:2])
                                else:
                                    # SDE trajectory: use add_noise + fresh random noise
                                    noisy_next = self.scheduler.add_noise(
                                        denoised_pred.flatten(0, 1),
                                        torch.randn_like(denoised_pred.flatten(0, 1)),
                                        next_timestep * torch.ones(
                                            [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
                                    ).unflatten(0, denoised_pred.shape[:2])
                                
                                context_timestep = torch.ones_like(timestep) * next_timestep
                                cache_input = noisy_next

                            # 3. Update KV Cache
                            self.generator(
                                noisy_image_or_video=cache_input,
                                conditional_dict=conditional_dict,
                                timestep=context_timestep,
                                kv_cache=self.kv_cache1,
                                crossattn_cache=self.crossattn_cache,
                                current_start=start_frame * self.frame_seq_length,
                            )

                    # After each timestep is done, reset KV cache so the next timestep can rebuild it
                    if not is_last_step:
                        reset_kv_cache()
                        recache_initial_latent()

                    if profile:
                        step_end.record()
                        torch.cuda.synchronize()
                        step_time = step_start.elapsed_time(step_end)
                        step_times.append(("timestep_first", step_index, step_time))

        if profile:
            diffusion_end.record()
            torch.cuda.synchronize()
            diffusion_time = diffusion_start.elapsed_time(diffusion_end)
            init_time = init_start.elapsed_time(init_end)
            vae_start.record()

        # Step 5: Decode the output
        video = self.vae.decode_to_pixel(output, use_cache=False)
        video = (video * 0.5 + 0.5).clamp(0, 1)

        if profile:
            vae_end.record()
            torch.cuda.synchronize()
            vae_time = vae_start.elapsed_time(vae_end)
            total_time = init_time + diffusion_time + vae_time

            print(f"Profiling results (hybrid mode: frame_first_steps={frame_first_steps}, early_return_step={early_return_step}):")
            print(f"  - Initialization/caching time: {init_time:.2f} ms ({100 * init_time / total_time:.2f}%)")
            print(f"  - Diffusion generation time: {diffusion_time:.2f} ms ({100 * diffusion_time / total_time:.2f}%)")
            for mode, idx, t_time in step_times:
                if mode == "frame_first":
                    print(f"    - [Frame-first] Block {idx} time: {t_time:.2f} ms ({100 * t_time / diffusion_time:.2f}% of diffusion)")
                else:
                    print(f"    - [Timestep-first] Step {idx} (t={self.denoising_step_list[idx].item()}) time: {t_time:.2f} ms ({100 * t_time / diffusion_time:.2f}% of diffusion)")
            print(f"  - VAE decoding time: {vae_time:.2f} ms ({100 * vae_time / total_time:.2f}%)")
            print(f"  - Total time: {total_time:.2f} ms")

        if return_latents:
            return video, output
        else:
            return video

    def inference_hybrid_block0(
        self,
        noise: torch.Tensor,
        text_prompts: List[str],
        initial_latent: Optional[torch.Tensor] = None,
        return_latents: bool = False,
        low_memory: bool = False,
        num_frame_first_blocks: int = 1,
    ) -> torch.Tensor:
        """
        Hybrid inference: first N blocks use frame-first order (fully denoised sequentially),
        remaining blocks use timestep-first order (HiAR-style hierarchical denoising).

        This tests the hypothesis that the first block benefits from frame-first denoising
        (no noisy context available anyway) while later blocks benefit from timestep-first
        (hierarchical denoising with matched noise levels).

        KV Cache Strategy:
        - Phase 1 (frame-first blocks): Each block is fully denoised through all steps,
          then its clean output is cached with t=context_noise (same as standard frame-first).
        - Phase 2 (timestep-first blocks): At each timestep, KV cache is reset and rebuilt:
          * Frame-first blocks: re-cached with clean latent at t=context_noise
          * Previous timestep-first blocks: cached with noisy latent at current noise level
          This matches HiAR's noisy context principle for the timestep-first portion.

        Inputs:
            noise: (B, num_frames, C, H, W)
            text_prompts: list of text prompts
            initial_latent: optional conditioning frames
            return_latents: whether to return latents
            low_memory: low GPU memory mode
            num_frame_first_blocks: number of leading blocks to denoise frame-first (default=1)
        """
        batch_size, num_frames, num_channels, height, width = noise.shape
        self._update_runtime_cache_spec(height, width)
        if not self.independent_first_frame or (self.independent_first_frame and initial_latent is not None):
            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block
        else:
            assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block
        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames
        num_denoising_steps = len(self.denoising_step_list)

        conditional_dict = self.text_encoder(text_prompts=text_prompts)

        if low_memory:
            gpu_memory_preservation = get_cuda_free_memory_gb(noise.device) + 5
            move_model_to_device_with_memory_preservation(
                self.text_encoder, target_device=noise.device,
                preserved_memory_gb=gpu_memory_preservation)

        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device, dtype=noise.dtype)

        # Step 1: Initialize KV cache
        if self.kv_cache1 is None:
            self._initialize_kv_cache(
                batch_size=batch_size, dtype=noise.dtype, device=noise.device)
            self._initialize_crossattn_cache(
                batch_size=batch_size, dtype=noise.dtype, device=noise.device)
        else:
            for block_index in range(self.num_transformer_blocks):
                self.crossattn_cache[block_index]["is_init"] = False
            for block_index in range(len(self.kv_cache1)):
                self.kv_cache1[block_index]["global_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                self.kv_cache1[block_index]["local_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)

        # Step 2: Cache initial_latent (if any)
        current_start_frame = 0
        num_input_blocks = 0
        if initial_latent is not None:
            timestep = torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0
            if self.independent_first_frame:
                assert (num_input_frames - 1) % self.num_frame_per_block == 0
                num_input_blocks = (num_input_frames - 1) // self.num_frame_per_block
                output[:, :1] = initial_latent[:, :1]
                self.generator(
                    noisy_image_or_video=initial_latent[:, :1],
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                )
                current_start_frame += 1
            else:
                assert num_input_frames % self.num_frame_per_block == 0
                num_input_blocks = num_input_frames // self.num_frame_per_block

            for _ in range(num_input_blocks):
                current_ref_latents = \
                    initial_latent[:, current_start_frame:current_start_frame + self.num_frame_per_block]
                output[:, current_start_frame:current_start_frame + self.num_frame_per_block] = current_ref_latents
                self.generator(
                    noisy_image_or_video=current_ref_latents,
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                )
                current_start_frame += self.num_frame_per_block

        # Step 3: Prepare block info
        all_num_frames = [self.num_frame_per_block] * num_blocks
        if self.independent_first_frame and initial_latent is None:
            all_num_frames = [1] + all_num_frames

        block_start_frames = []
        temp_start = current_start_frame
        for nf in all_num_frames:
            block_start_frames.append(temp_start)
            temp_start += nf

        # Initialize noisy inputs for each block
        noisy_inputs = []
        for i, (start_frame, nf) in enumerate(zip(block_start_frames, all_num_frames)):
            noisy_input = noise[:, start_frame - num_input_frames:start_frame + nf - num_input_frames]
            noisy_inputs.append(noisy_input)

        # Store clean outputs for frame-first blocks (used for KV re-caching in phase 2)
        frame_first_clean = [None] * len(all_num_frames)

        # Helper: reset KV cache
        def reset_kv_cache():
            for block_idx in range(self.num_transformer_blocks):
                self.crossattn_cache[block_idx]["is_init"] = False
            for block_idx in range(len(self.kv_cache1)):
                self.kv_cache1[block_idx]["global_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                self.kv_cache1[block_idx]["local_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)

        # Helper: re-cache initial_latent
        def recache_initial_latent():
            if initial_latent is not None:
                temp_sf = 0
                init_ts = torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0
                if self.independent_first_frame:
                    self.generator(
                        noisy_image_or_video=initial_latent[:, :1],
                        conditional_dict=conditional_dict,
                        timestep=init_ts * 0,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=temp_sf * self.frame_seq_length,
                    )
                    temp_sf += 1
                for _ in range(num_input_blocks):
                    current_ref_latents = \
                        initial_latent[:, temp_sf:temp_sf + self.num_frame_per_block]
                    self.generator(
                        noisy_image_or_video=current_ref_latents,
                        conditional_dict=conditional_dict,
                        timestep=init_ts * 0,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=temp_sf * self.frame_seq_length,
                    )
                    temp_sf += self.num_frame_per_block

        # Helper: re-cache the frame-first blocks' clean outputs into KV cache
        def recache_frame_first_blocks():
            for bi in range(num_frame_first_blocks):
                sf = block_start_frames[bi]
                nf = all_num_frames[bi]
                ctx_ts = torch.ones(
                    [batch_size, nf], device=noise.device,
                    dtype=torch.int64) * self.args.context_noise
                self.generator(
                    noisy_image_or_video=frame_first_clean[bi],
                    conditional_dict=conditional_dict,
                    timestep=ctx_ts,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=sf * self.frame_seq_length,
                )

        # Clamp num_frame_first_blocks
        total_blocks = len(all_num_frames)
        num_frame_first_blocks = min(num_frame_first_blocks, total_blocks)

        print(f"Hybrid-Block0 inference: {num_frame_first_blocks} block(s) frame-first, "
              f"{total_blocks - num_frame_first_blocks} block(s) timestep-first")

        # ========== Phase 1: Frame-first denoising for first num_frame_first_blocks blocks ==========
        for block_index in range(num_frame_first_blocks):
            start_frame = block_start_frames[block_index]
            current_num_frames = all_num_frames[block_index]
            noisy_input = noisy_inputs[block_index]

            for step_index in range(num_denoising_steps):
                current_timestep = self.denoising_step_list[step_index]
                timestep = torch.ones(
                    [batch_size, current_num_frames],
                    device=noise.device, dtype=torch.int64) * current_timestep

                _, denoised_pred = self.generator(
                    noisy_image_or_video=noisy_input,
                    conditional_dict=conditional_dict,
                    timestep=timestep,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=start_frame * self.frame_seq_length,
                )

                if step_index < num_denoising_steps - 1:
                    next_timestep = self.denoising_step_list[step_index + 1]
                    if self.use_ode_trajectory:
                        raise NotImplementedError("ODE trajectory not needed for this test")
                    else:
                        noisy_input = self.scheduler.add_noise(
                            denoised_pred.flatten(0, 1),
                            torch.randn_like(denoised_pred.flatten(0, 1)),
                            next_timestep * torch.ones(
                                [batch_size * current_num_frames],
                                device=noise.device, dtype=torch.long)
                        ).unflatten(0, denoised_pred.shape[:2])

            # Store clean output and record in output tensor
            frame_first_clean[block_index] = denoised_pred
            output[:, start_frame:start_frame + current_num_frames] = denoised_pred

            # Cache clean context for subsequent frame-first blocks
            context_timestep = torch.ones(
                [batch_size, current_num_frames],
                device=noise.device, dtype=torch.int64) * self.args.context_noise
            self.generator(
                noisy_image_or_video=denoised_pred,
                conditional_dict=conditional_dict,
                timestep=context_timestep,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=start_frame * self.frame_seq_length,
            )

        # ========== Phase 2: Timestep-first denoising for remaining blocks ==========
        if num_frame_first_blocks < total_blocks:
            for step_index in range(num_denoising_steps):
                current_timestep = self.denoising_step_list[step_index]
                is_last_step = (step_index == num_denoising_steps - 1)

                # Reset KV cache and rebuild: initial_latent + frame-first blocks
                reset_kv_cache()
                recache_initial_latent()
                recache_frame_first_blocks()

                # Inner loop: iterate over timestep-first blocks
                for block_index in tqdm(
                    range(num_frame_first_blocks, total_blocks),
                    desc=f"Step {step_index} (t={current_timestep.item():.0f})",
                    leave=False,
                ):
                    start_frame = block_start_frames[block_index]
                    current_num_frames = all_num_frames[block_index]
                    noisy_input = noisy_inputs[block_index]

                    timestep = torch.ones(
                        [batch_size, current_num_frames],
                        device=noise.device, dtype=torch.int64) * current_timestep

                    flow_pred, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=start_frame * self.frame_seq_length,
                    )

                    if not is_last_step:
                        next_timestep = self.denoising_step_list[step_index + 1]
                        if self.use_ode_trajectory:
                            raise NotImplementedError("ODE trajectory not needed for this test")
                        else:
                            noisy_inputs[block_index] = self.scheduler.add_noise(
                                denoised_pred.flatten(0, 1),
                                torch.randn_like(denoised_pred.flatten(0, 1)),
                                next_timestep * torch.ones(
                                    [batch_size * current_num_frames],
                                    device=noise.device, dtype=torch.long)
                            ).unflatten(0, denoised_pred.shape[:2])

                        # Cache noisy context (HiAR-style)
                        if self.always_clean_context:
                            context_timestep = torch.ones_like(timestep) * self.args.context_noise
                            cache_input = denoised_pred
                        else:
                            context_timestep = torch.ones_like(timestep) * next_timestep
                            cache_input = noisy_inputs[block_index]
                        self.generator(
                            noisy_image_or_video=cache_input,
                            conditional_dict=conditional_dict,
                            timestep=context_timestep,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=start_frame * self.frame_seq_length,
                        )
                    else:
                        # Last step: record output and cache clean context
                        output[:, start_frame:start_frame + current_num_frames] = denoised_pred
                        context_timestep = torch.ones_like(timestep) * self.args.context_noise
                        self.generator(
                            noisy_image_or_video=denoised_pred,
                            conditional_dict=conditional_dict,
                            timestep=context_timestep,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=start_frame * self.frame_seq_length,
                        )

        # Step 4: Decode
        video = self.vae.decode_to_pixel(output, use_cache=False)
        video = (video * 0.5 + 0.5).clamp(0, 1)

        if return_latents:
            return video, output
        else:
            return video

    def inference_pipeline_parallel(
        self,
        noise: torch.Tensor,
        text_prompts: List[str],
        initial_latent: Optional[torch.Tensor] = None,
        return_latents: bool = False,
        profile: bool = False,
        low_memory: bool = False,
    ) -> torch.Tensor:
        """
        Pipeline-parallel pure Timestep-First inference.

        Core idea: For S denoising steps and B frame blocks, the original timestep-first mode
        sequentially executes S*B model forward passes. Leveraging the causal dependency structure,
        when step k processes block j, step k+1 can simultaneously process block j-1 (since j-1
        was already completed at step k), forming a diagonal pipeline.

        On a single GPU, we use N = num_denoising_steps independent KV caches to simulate N
        pipeline stages. Each "micro-step" traverses (step, block) pairs diagonally, so that
        different stages can share GPU compute at the same moment (sequential execution of
        micro-steps ensures correct data dependencies).

        Although there is no true parallel speedup on a single GPU (model forward passes are
        still sequential), the core value of this function is:
        1. Providing correct scheduling logic and data flow templates for multi-GPU pipeline parallelism
        2. Reducing KV cache rebuild count (each stage maintains its own KV cache, no need to rebuild per step)
        3. Producing final results for earlier blocks sooner (streaming output friendly)

        Strictly equivalent to the timestep-first mode of the original inference_hybrid:
        - Each block undergoes the same denoising step sequence
        - KV cache update logic is consistent (clean/noisy context)
        - Noise addition / ODE step logic is consistent

        Inputs:
            noise: (B, num_frames, C, H, W) input noise
            text_prompts: list of text prompts
            initial_latent: optional initial frame latent
            return_latents: whether to also return the latent
            profile: whether to perform performance profiling
            low_memory: low GPU memory mode
        Outputs:
            video: (B, num_frames, C, H, W) generated video, range [0, 1]
        """
        batch_size, num_frames, num_channels, height, width = noise.shape
        self._update_runtime_cache_spec(height, width)
        
        # ---- 1. Compute number of blocks ----
        if not self.independent_first_frame or (self.independent_first_frame and initial_latent is not None):
            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block
        else:
            assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block
        
        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames
        num_stages = len(self.denoising_step_list)  # number of pipeline stages = number of denoising steps
        
        conditional_dict = self.text_encoder(text_prompts=text_prompts)
        
        if low_memory:
            from utils.memory import get_cuda_free_memory_gb, move_model_to_device_with_memory_preservation
            gpu_memory_preservation = get_cuda_free_memory_gb(noise.device) + 5
            move_model_to_device_with_memory_preservation(
                self.text_encoder, target_device=noise.device, preserved_memory_gb=gpu_memory_preservation)
        
        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device, dtype=noise.dtype
        )
        
        if profile:
            prof_start = torch.cuda.Event(enable_timing=True)
            prof_end = torch.cuda.Event(enable_timing=True)
            micro_times = []
            prof_start.record()
        
        # ---- 2. Initialize independent KV cache for each stage ----
        # stage_kv_caches[s] = kv_cache for stage s
        # stage_crossattn_caches[s] = crossattn_cache for stage s
        stage_kv_caches = []
        stage_crossattn_caches = []
        
        if self.local_attn_size != -1:
            kv_cache_size = self.local_attn_size * self.frame_seq_length
        else:
            kv_cache_size = 32760
        
        for _ in range(num_stages):
            kv_cache = []
            for __ in range(self.num_transformer_blocks):
                kv_cache.append({
                    "k": torch.zeros([batch_size, kv_cache_size, self.num_attention_heads, self.attention_head_dim],
                                     dtype=noise.dtype, device=noise.device),
                    "v": torch.zeros([batch_size, kv_cache_size, self.num_attention_heads, self.attention_head_dim],
                                     dtype=noise.dtype, device=noise.device),
                    "global_end_index": torch.tensor([0], dtype=torch.long, device=noise.device),
                    "local_end_index": torch.tensor([0], dtype=torch.long, device=noise.device),
                })
            stage_kv_caches.append(kv_cache)
            
            crossattn_cache = []
            for __ in range(self.num_transformer_blocks):
                crossattn_cache.append({
                    "k": torch.zeros([batch_size, self.text_context_len, self.num_attention_heads, self.attention_head_dim],
                                     dtype=noise.dtype, device=noise.device),
                    "v": torch.zeros([batch_size, self.text_context_len, self.num_attention_heads, self.attention_head_dim],
                                     dtype=noise.dtype, device=noise.device),
                    "is_init": False,
                })
            stage_crossattn_caches.append(crossattn_cache)
        
        # ---- 3. Helper: reset KV cache for a given stage ----
        def reset_stage_cache(stage_idx):
            for blk in range(self.num_transformer_blocks):
                stage_crossattn_caches[stage_idx][blk]["is_init"] = False
            for blk in range(len(stage_kv_caches[stage_idx])):
                stage_kv_caches[stage_idx][blk]["global_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                stage_kv_caches[stage_idx][blk]["local_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
        
        # ---- 4. Helper: cache initial_latent into a given stage's KV cache ----
        def cache_initial_latent_to_stage(stage_idx):
            if initial_latent is None:
                return
            kv = stage_kv_caches[stage_idx]
            ca = stage_crossattn_caches[stage_idx]
            temp_start = 0
            init_ts = torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0
            
            if self.independent_first_frame:
                self.generator(
                    noisy_image_or_video=initial_latent[:, :1],
                    conditional_dict=conditional_dict,
                    timestep=init_ts * 0,
                    kv_cache=kv, crossattn_cache=ca,
                    current_start=temp_start * self.frame_seq_length,
                )
                temp_start += 1
            
            _num_input_blocks = ((num_input_frames - (1 if self.independent_first_frame else 0))
                                 // self.num_frame_per_block)
            for _ in range(_num_input_blocks):
                ref = initial_latent[:, temp_start:temp_start + self.num_frame_per_block]
                self.generator(
                    noisy_image_or_video=ref,
                    conditional_dict=conditional_dict,
                    timestep=init_ts * 0,
                    kv_cache=kv, crossattn_cache=ca,
                    current_start=temp_start * self.frame_seq_length,
                )
                temp_start += self.num_frame_per_block
        
        # ---- 5. Initialize initial_latent cache for all stages ----
        if initial_latent is not None:
            output[:, :num_input_frames] = initial_latent
        
        for s in range(num_stages):
            cache_initial_latent_to_stage(s)
        
        # ---- 6. Prepare frame block information ----
        all_num_frames_list = [self.num_frame_per_block] * num_blocks
        if self.independent_first_frame and initial_latent is None:
            all_num_frames_list = [1] + all_num_frames_list
        
        num_total_blocks = len(all_num_frames_list)
        
        # Compute the start frame for each block
        block_start_frames = []
        temp = num_input_frames if initial_latent is not None else 0
        if self.independent_first_frame and initial_latent is not None:
            temp = num_input_frames
        elif self.independent_first_frame and initial_latent is None:
            temp = 0
        else:
            temp = num_input_frames
        for nf in all_num_frames_list:
            block_start_frames.append(temp)
            temp += nf
        
        # Initialize noisy input for each block
        noisy_inputs = []
        for i, (sf, nf) in enumerate(zip(block_start_frames, all_num_frames_list)):
            ni = noise[:, sf - num_input_frames:sf + nf - num_input_frames]
            noisy_inputs.append(ni)
        
        # flow_preds[block_idx] stores the most recent flow_pred (used for ODE step)
        flow_preds = [None] * num_total_blocks
        # denoised_preds[block_idx] stores the most recent denoised_pred
        denoised_preds = [None] * num_total_blocks
        # block_completed_step[block_idx] = highest completed step index (-1 means not started)
        block_completed_step = [-1] * num_total_blocks
        
        # ---- 7. Pipeline scheduling: diagonal traversal of (stage, block) ----
        # Total of num_stages + num_total_blocks - 1 micro-steps (diagonal wave)
        # In micro-step d, all (stage, block) pairs satisfying stage + block == d can execute in parallel
        # On a single GPU we execute them sequentially, but data dependencies are correctly satisfied
        
        num_diagonals = num_stages + num_total_blocks - 1
        
        for diag in tqdm(range(num_diagonals), desc="Pipeline parallel inference"):
            # Collect (stage, block) pairs on the current diagonal
            pairs = []
            for stage in range(num_stages):
                block = diag - stage
                if 0 <= block < num_total_blocks:
                    pairs.append((stage, block))
            
            for stage, block_idx in pairs:
                if profile:
                    ms = torch.cuda.Event(enable_timing=True)
                    me = torch.cuda.Event(enable_timing=True)
                    ms.record()
                
                start_frame = block_start_frames[block_idx]
                cur_nf = all_num_frames_list[block_idx]
                current_timestep = self.denoising_step_list[stage]
                kv = stage_kv_caches[stage]
                ca = stage_crossattn_caches[stage]
                is_last_stage = (stage == num_stages - 1)
                
                # Get the current block's noisy input
                noisy_input = noisy_inputs[block_idx]
                
                timestep = torch.ones(
                    [batch_size, cur_nf], device=noise.device, dtype=torch.int64
                ) * current_timestep
                
                # ---- 7a. Denoising forward pass ----
                flow_pred, denoised_pred = self.generator(
                    noisy_image_or_video=noisy_input,
                    conditional_dict=conditional_dict,
                    timestep=timestep,
                    kv_cache=kv, crossattn_cache=ca,
                    current_start=start_frame * self.frame_seq_length,
                )
                
                flow_preds[block_idx] = flow_pred
                denoised_preds[block_idx] = denoised_pred
                
                if not is_last_stage:
                    # ---- 7b. Add noise to the next timestep (prepare input for the next stage) ----
                    next_timestep = self.denoising_step_list[stage + 1]
                    if self.use_ode_trajectory:
                        noisy_inputs[block_idx] = self.scheduler.step(
                            flow_pred.flatten(0, 1),
                            timestep.flatten(0, 1),
                            noisy_input.flatten(0, 1),
                            target_timestep=next_timestep
                        ).unflatten(0, denoised_pred.shape[:2])
                    else:
                        noisy_inputs[block_idx] = self.scheduler.add_noise(
                            denoised_pred.flatten(0, 1),
                            torch.randn_like(denoised_pred.flatten(0, 1)),
                            next_timestep * torch.ones(
                                [batch_size * cur_nf], device=noise.device, dtype=torch.long)
                        ).unflatten(0, denoised_pred.shape[:2])
                    
                    # ---- 7c. Update the current stage's KV cache ----
                    if self.always_clean_context:
                        ctx_ts = torch.ones_like(timestep) * self.args.context_noise
                        cache_input = denoised_pred
                    else:
                        ctx_ts = torch.ones_like(timestep) * next_timestep
                        cache_input = noisy_inputs[block_idx]
                    
                    self.generator(
                        noisy_image_or_video=cache_input,
                        conditional_dict=conditional_dict,
                        timestep=ctx_ts,
                        kv_cache=kv, crossattn_cache=ca,
                        current_start=start_frame * self.frame_seq_length,
                    )
                else:
                    # ---- 7d. Last stage: record final output ----
                    output[:, start_frame:start_frame + cur_nf] = denoised_pred
                    
                    # Update KV cache with clean context
                    ctx_ts = torch.ones_like(timestep) * self.args.context_noise
                    self.generator(
                        noisy_image_or_video=denoised_pred,
                        conditional_dict=conditional_dict,
                        timestep=ctx_ts,
                        kv_cache=kv, crossattn_cache=ca,
                        current_start=start_frame * self.frame_seq_length,
                    )
                
                block_completed_step[block_idx] = stage
                
                if profile:
                    me.record()
                    torch.cuda.synchronize()
                    micro_times.append((diag, stage, block_idx, ms.elapsed_time(me)))
            
            # ---- 7e. Diagonal boundary: reset KV cache for stages that have completed all blocks ----
            # After a stage finishes its last block, it needs to prepare for the next wave
            # But in the pipeline, each stage only passes through all blocks once, no reset needed
            # (Unlike the original timestep-first mode which resets at every step)
        
        if profile:
            prof_end.record()
            torch.cuda.synchronize()
            total_ms = prof_start.elapsed_time(prof_end)
            print(f"\nPipeline parallel inference performance analysis:")
            print(f"  - Total time: {total_ms:.2f} ms")
            print(f"  - Number of diagonals: {num_diagonals}, total micro-steps: {sum(1 for _ in micro_times)}")
            print(f"  - Denoising steps: {num_stages}, frame blocks: {num_total_blocks}")
            # Per-stage statistics
            stage_total = {}
            for _, s, _, t in micro_times:
                stage_total[s] = stage_total.get(s, 0.0) + t
            for s in sorted(stage_total.keys()):
                print(f"  - Stage {s} (t={self.denoising_step_list[s].item():.0f}): {stage_total[s]:.2f} ms")
        
        # ---- 8. VAE decoding ----
        video = self.vae.decode_to_pixel(output, use_cache=False)
        video = (video * 0.5 + 0.5).clamp(0, 1)
        
        if return_latents:
            return video, output
        else:
            return video
