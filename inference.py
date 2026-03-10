import argparse
import torch
import os
import imageio
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm
from einops import rearrange
import torch.distributed as dist
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pipeline import CausalInferencePipeline
from utils.dataset import TextDataset
from utils.misc import set_seed
from utils.memory import get_cuda_free_memory_gb, DynamicSwapInstaller


def _clean_state_dict_keys(state_dict):
    return {
        k.replace("_fsdp_wrapped_module.", "")
         .replace("_checkpoint_wrapped_module.", "")
         .replace("_orig_mod.", ""): v
        for k, v in state_dict.items()
    }


def _extract_generator_state_dict(checkpoint, use_ema=False):
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)}")

    preferred_keys = ["generator_ema", "generator"] if use_ema else ["generator", "model", "generator_ema"]
    for key in preferred_keys:
        if key in checkpoint and isinstance(checkpoint[key], dict):
            if use_ema and key != "generator_ema":
                print(f"Warning: EMA key not found in checkpoint, falling back to '{key}'")
            return checkpoint[key]

    if all(isinstance(k, str) for k in checkpoint.keys()):
        return checkpoint

    available_keys = ", ".join(map(str, checkpoint.keys()))
    raise KeyError(
        "Unable to locate generator weights in checkpoint. "
        f"Available top-level keys: {available_keys}"
    )


parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, help="Path to the config file")
parser.add_argument("--checkpoint_path", type=str, help="Path to the checkpoint file")
parser.add_argument("--data_path", type=str, default=None, help="Path to the prompt text file")
parser.add_argument("--prompt", type=str, default=None, help="Single prompt to run inference on")
parser.add_argument("--extended_prompt_path", type=str, default=None, help="Path to the extended prompt file")
parser.add_argument("--extended_prompt", type=str, default=None, help="Single extended prompt paired with --prompt")
parser.add_argument("--output_folder", type=str, default=None, help="Output folder")
parser.add_argument("--output_path", type=str, default=None, help="Output video path for single-prompt inference")
parser.add_argument("--num_output_frames", type=int, default=21,
                    help="Number of latent frames to generate")
parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA parameters")
parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to generate per prompt")
parser.add_argument("--save_with_index", action="store_true",
                    help="Save the video using the index instead of prompt as the filename")
parser.add_argument("--inference_method", type=str, default="frame_first",
                    choices=["frame_first", "timestep_first", "hybrid_block0"],
                    help="Denoising order: frame_first (Self-Forcing), timestep_first (HiAR), "
                         "hybrid_block0 (first block frame-first, rest timestep-first), "
)
parser.add_argument("--num_frame_first_blocks", type=int, default=1,
                    help="Number of leading blocks to denoise frame-first in hybrid_block0 mode")
args = parser.parse_args()

if (args.prompt is None) == (args.data_path is None):
    raise ValueError("Specify exactly one of --prompt or --data_path")
if args.extended_prompt is not None and args.prompt is None:
    raise ValueError("--extended_prompt requires --prompt")
if args.extended_prompt_path is not None and args.data_path is None:
    raise ValueError("--extended_prompt_path requires --data_path")
if args.output_path is not None and args.prompt is None:
    raise ValueError("--output_path requires --prompt")
if (args.output_folder is None) == (args.output_path is None):
    raise ValueError("Specify exactly one of --output_folder or --output_path")

# Initialize distributed inference
if "LOCAL_RANK" in os.environ:
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    world_size = dist.get_world_size()
    set_seed(args.seed + local_rank)
else:
    device = torch.device("cuda")
    local_rank = 0
    world_size = 1
    set_seed(args.seed)

print(f'Free VRAM {get_cuda_free_memory_gb(device)} GB')
low_memory = get_cuda_free_memory_gb(device) < 40

torch.set_grad_enabled(False)

config = OmegaConf.load(args.config_path)
default_config = OmegaConf.load("configs/default_config.yaml")
config = OmegaConf.merge(default_config, config)

num_frame_per_block = getattr(config, "num_frame_per_block", 1)
independent_first_frame = getattr(config, "independent_first_frame", False)
requested_num_output_frames = args.num_output_frames
if independent_first_frame:
    remainder = (requested_num_output_frames - 1) % num_frame_per_block
else:
    remainder = requested_num_output_frames % num_frame_per_block
if remainder != 0:
    aligned_num_output_frames = requested_num_output_frames + (num_frame_per_block - remainder)
    print(
        f"num_output_frames ({requested_num_output_frames}) is not divisible by num_frame_per_block "
        f"({num_frame_per_block}); rounding up to {aligned_num_output_frames}"
    )
    args.num_output_frames = aligned_num_output_frames

# Initialize pipeline
pipeline = CausalInferencePipeline(config, device=device)

if args.checkpoint_path:
    checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
    raw_sd = _extract_generator_state_dict(checkpoint, use_ema=args.use_ema)
    clean_sd = _clean_state_dict_keys(raw_sd)
    pipeline.generator.load_state_dict(clean_sd)

pipeline = pipeline.to(dtype=torch.bfloat16)
if low_memory:
    DynamicSwapInstaller.install_model(pipeline.text_encoder, device=device)
else:
    pipeline.text_encoder.to(device=device)
pipeline.generator.to(device=device)
pipeline.vae.to(device=device)

# Enable local attention for long videos to avoid KV cache overflow
if args.num_output_frames > 21:
    local_attn = 21
    print(f"num_output_frames ({args.num_output_frames}) > 21, enabling local attention with window size {local_attn}")
    pipeline.local_attn_size = local_attn
    pipeline.generator.model.local_attn_size = local_attn
    for block in pipeline.generator.model.blocks:
        block.self_attn.local_attn_size = local_attn
        block.self_attn.max_attention_size = local_attn * 1560

# Create dataset
if args.prompt is not None:
    dataset = [{"prompts": args.prompt, "idx": 0}]
    if args.extended_prompt is not None:
        dataset[0]["extended_prompts"] = args.extended_prompt
    if dist.is_initialized() and local_rank != 0:
        dataset = []
    num_prompts = 1
else:
    dataset = TextDataset(prompt_path=args.data_path, extended_prompt_path=args.extended_prompt_path)
    num_prompts = len(dataset)
print(f"Number of prompts: {num_prompts}")

if dist.is_initialized() and args.prompt is None:
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=True)
else:
    sampler = SequentialSampler(dataset)
dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0, drop_last=False)

# Create output directory (only on main process to avoid race conditions)
if local_rank == 0:
    if args.output_path is not None:
        output_dir = os.path.dirname(args.output_path) or "."
        os.makedirs(output_dir, exist_ok=True)
    else:
        os.makedirs(args.output_folder, exist_ok=True)

if dist.is_initialized():
    dist.barrier()

for i, batch_data in tqdm(enumerate(dataloader), disable=(local_rank != 0)):
    idx = batch_data['idx'].item()

    if isinstance(batch_data, dict):
        batch = batch_data
    elif isinstance(batch_data, list):
        batch = batch_data[0]

    prompt = batch['prompts'][0]
    extended_prompt = batch['extended_prompts'][0] if 'extended_prompts' in batch else None
    if extended_prompt is not None:
        prompts = [extended_prompt] * args.num_samples
    else:
        prompts = [prompt] * args.num_samples

    sampled_noise = torch.randn(
        [args.num_samples, args.num_output_frames, 16, 60, 104], device=device, dtype=torch.bfloat16
    )

    if args.inference_method == "timestep_first":
        video, latents = pipeline.inference_hybrid(
            noise=sampled_noise,
            text_prompts=prompts,
            return_latents=True,
            low_memory=low_memory,
            frame_first_steps=0,
        )
    elif args.inference_method == "hybrid_block0":
        video, latents = pipeline.inference_hybrid_block0(
            noise=sampled_noise,
            text_prompts=prompts,
            return_latents=True,
            low_memory=low_memory,
            num_frame_first_blocks=args.num_frame_first_blocks,
        )
    else:
        video, latents = pipeline.inference(
            noise=sampled_noise,
            text_prompts=prompts,
            return_latents=True,
            low_memory=low_memory,
        )
    current_video = rearrange(video, 'b t c h w -> b t h w c').cpu()
    video_out = 255.0 * current_video

    # Clear VAE cache
    pipeline.vae.model.clear_cache()

    if idx < num_prompts:
        model = "regular" if not args.use_ema else "ema"
        for seed_idx in range(args.num_samples):
            if args.output_path is not None:
                output_root, output_ext = os.path.splitext(args.output_path)
                output_ext = output_ext or '.mp4'
                if args.num_samples == 1:
                    output_path = f'{output_root}{output_ext}'
                else:
                    output_path = f'{output_root}-{seed_idx}_{model}{output_ext}'
            elif args.save_with_index:
                output_path = os.path.join(args.output_folder, f'{idx}-{seed_idx}_{model}.mp4')
            else:
                output_path = os.path.join(args.output_folder, f'{prompt[:100]}-{seed_idx}.mp4')
            video_np = video_out[seed_idx].clamp(0, 255).to(torch.uint8).numpy()
            imageio.mimwrite(output_path, video_np, fps=16, quality=8)

if dist.is_initialized():
    dist.barrier()
    dist.destroy_process_group()
