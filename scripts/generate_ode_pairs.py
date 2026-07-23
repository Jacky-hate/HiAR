"""
Generate ODE trajectory pairs for forward-KL regularization.

Samples ODE trajectories from the teacher model (Wan2.1) and saves
intermediate latents at specified timestep indices. These pairs are
used during HiAR training as targets for the forward-KL regularizer.

Usage (multi-GPU with torchrun):
    torchrun --nproc_per_node=8 scripts/generate_ode_pairs.py \
        --output_folder data/ode_pairs \
        --caption_path data/vidprom_filtered_extended.txt \
        --guidance_scale 6.0 \
        --num_inference_steps 50 \
        --num_prompts 20000

The saved points are interpolated at the student's exact four-step sigmas.
"""
from utils.distributed import launch_distributed_job
from utils.scheduler import FlowMatchScheduler
from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder
from utils.dataset import TextDataset
from utils.pd_schedule import interpolate_trajectory, shifted_student_sigmas
import torch.distributed as dist
from tqdm import tqdm
import argparse
import json
import torch
import math
import os
import time
import logging


def setup_logger(rank, output_folder):
    logger = logging.getLogger(f"rank_{rank}")
    logger.setLevel(logging.INFO)
    if rank == 0:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter("[%(asctime)s][Rank %(name)s] %(message)s")
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    os.makedirs(output_folder, exist_ok=True)
    fh = logging.FileHandler(os.path.join(output_folder, f"rank_{rank}.log"))
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(asctime)s][Rank %(name)s] %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


NEGATIVE_PROMPT = (
    "\u8272\u8c03\u8273\u4e3d\uff0c\u8fc7\u66dd\uff0c\u9759\u6001\uff0c"
    "\u7ec6\u8282\u6a21\u7cca\u4e0d\u6e05\uff0c\u5b57\u5e55\uff0c\u98ce\u683c\uff0c"
    "\u4f5c\u54c1\uff0c\u753b\u4f5c\uff0c\u753b\u9762\uff0c\u9759\u6b62\uff0c"
    "\u6574\u4f53\u53d1\u7070\uff0c\u6700\u5dee\u8d28\u91cf\uff0c\u4f4e\u8d28\u91cf\uff0c"
    "JPEG\u538b\u7f29\u6b8b\u7559\uff0c\u4e11\u964b\u7684\uff0c\u6b8b\u7f3a\u7684\uff0c"
    "\u591a\u4f59\u7684\u624b\u6307\uff0c\u753b\u5f97\u4e0d\u597d\u7684\u624b\u90e8\uff0c"
    "\u753b\u5f97\u4e0d\u597d\u7684\u8138\u90e8\uff0c\u7578\u5f62\u7684\uff0c\u6bc1\u5bb9\u7684\uff0c"
    "\u5f62\u6001\u7578\u5f62\u7684\u80a2\u4f53\uff0c\u624b\u6307\u878d\u5408\uff0c"
    "\u9759\u6b62\u4e0d\u52a8\u7684\u753b\u9762\uff0c\u6742\u4e71\u7684\u80cc\u666f\uff0c"
    "\u4e09\u6761\u817f\uff0c\u80cc\u666f\u4eba\u5f88\u591a\uff0c\u5012\u7740\u8d70"
)


def init_model(device, model_name="Wan2.1-T2V-1.3B", num_inference_steps=48, timestep_shift=8.0):
    model = WanDiffusionWrapper(model_name=model_name, timestep_shift=timestep_shift).to(device).to(torch.float32)
    encoder = WanTextEncoder().to(device).to(torch.float32)
    model.model.requires_grad_(False)
    encoder.requires_grad_(False)

    scheduler = FlowMatchScheduler(
        shift=timestep_shift, sigma_min=0.0, extra_one_step=True)
    scheduler.set_timesteps(num_inference_steps=num_inference_steps, denoising_strength=1.0)
    scheduler.sigmas = scheduler.sigmas.to(device)

    unconditional_dict = encoder(text_prompts=[NEGATIVE_PROMPT])
    return model, encoder, scheduler, unconditional_dict


def get_completed_indices(output_folder):
    completed = set()
    if not os.path.exists(output_folder):
        return completed
    for fname in os.listdir(output_folder):
        if fname.endswith(".pt"):
            try:
                idx = int(fname.replace(".pt", ""))
                completed.add(idx)
            except ValueError:
                continue
    return completed


def main():
    parser = argparse.ArgumentParser(description="Generate ODE trajectory pairs for forward-KL")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--output_folder", type=str, required=True,
                        help="Output directory for ODE trajectory pairs")
    parser.add_argument("--caption_path", type=str, required=True,
                        help="Path to prompt text file (one prompt per line)")
    parser.add_argument("--guidance_scale", type=float, default=6.0,
                        help="Classifier-Free Guidance scale")
    parser.add_argument("--num_inference_steps", type=int, default=50,
                        help="Number of ODE sampling steps")
    parser.add_argument("--num_prompts", type=int, default=-1,
                        help="Number of prompts to use (-1 for all)")
    parser.add_argument(
        "--student_denoising_steps",
        type=str,
        default="1000,750,500,250,0",
        help="Nominal student timesteps whose exact sigmas are saved",
    )
    parser.add_argument(
        "--student_timestep_shift",
        type=float,
        default=5.0,
        help="Student flow-matching timestep shift",
    )
    parser.add_argument("--latent_shape", type=str, default="1,21,16,60,104",
                        help="Latent tensor shape (comma-separated)")
    parser.add_argument("--num_frames", type=int, default=21)
    parser.add_argument("--model_name", type=str, default="Wan2.1-T2V-1.3B")
    parser.add_argument("--timestep_shift", type=float, default=8.0)
    parser.add_argument("--resume", action="store_true",
                        help="Resume from previous run, skip completed prompts")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    launch_distributed_job()

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.cuda.current_device()

    logger = setup_logger(rank, args.output_folder)
    logger.info(f"Distributed init: rank={rank}, world_size={world_size}, device=cuda:{device}")

    latent_shape = list(map(int, args.latent_shape.split(",")))
    student_steps = list(map(int, args.student_denoising_steps.split(",")))
    aligned_sigmas = shifted_student_sigmas(
        student_steps, args.student_timestep_shift
    )
    num_frames = args.num_frames

    torch.manual_seed(args.seed + rank)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    logger.info("Loading model...")
    model, encoder, scheduler, unconditional_dict = init_model(
        device=device,
        model_name=args.model_name,
        num_inference_steps=args.num_inference_steps,
        timestep_shift=args.timestep_shift
    )
    logger.info("Model loaded")

    manifest = {
        "format_version": 2,
        "teacher_model": args.model_name,
        "teacher_timestep_shift": args.timestep_shift,
        "teacher_num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "student_timestep_shift": args.student_timestep_shift,
        "student_denoising_steps": student_steps,
        "aligned_sigmas": aligned_sigmas,
        "latent_shape": latent_shape,
        "dtype": "float32",
    }
    manifest_error = None
    if rank == 0:
        manifest_path = os.path.join(args.output_folder, "manifest.json")
        try:
            if os.path.isfile(manifest_path):
                with open(manifest_path, encoding="utf-8") as handle:
                    existing_manifest = json.load(handle)
                if existing_manifest != manifest:
                    raise RuntimeError(
                        "Existing manifest does not match requested ODE settings"
                    )
            else:
                temporary_manifest = manifest_path + ".tmp"
                with open(temporary_manifest, "w", encoding="utf-8") as handle:
                    json.dump(manifest, handle, indent=2, sort_keys=True)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary_manifest, manifest_path)
        except Exception as exc:
            manifest_error = f"{type(exc).__name__}: {exc}"
    payload = [manifest_error]
    dist.broadcast_object_list(payload, src=0)
    if payload[0] is not None:
        raise RuntimeError(f"Failed to initialize ODE manifest: {payload[0]}")
    dist.barrier()

    dataset = TextDataset(args.caption_path)
    total_prompts = len(dataset)
    if args.num_prompts > 0:
        total_prompts = min(args.num_prompts, total_prompts)
    logger.info(f"Total prompts: {total_prompts}")

    os.makedirs(args.output_folder, exist_ok=True)

    completed_indices = set()
    if args.resume:
        completed_indices = get_completed_indices(args.output_folder)
        logger.info(f"Resuming: {len(completed_indices)} already completed")

    num_iters = int(math.ceil(total_prompts / world_size))
    skipped = 0
    processed = 0
    failed = 0
    start_time = time.time()

    for index in tqdm(range(num_iters), disable=rank != 0, desc="ODE sampling"):
        prompt_index = index * world_size + rank
        if prompt_index >= total_prompts:
            continue

        if prompt_index in completed_indices:
            skipped += 1
            continue

        try:
            prompt_data = dataset[prompt_index]
            if isinstance(prompt_data, dict):
                prompt = prompt_data["prompts"]
            else:
                prompt = prompt_data

            conditional_dict = encoder(text_prompts=[prompt] if isinstance(prompt, str) else prompt)
            latents = torch.randn(latent_shape, dtype=torch.float32, device=device)
            noisy_input = []

            for progress_id, t in enumerate(scheduler.timesteps):
                timestep = t * torch.ones([1, num_frames], device=device, dtype=torch.float32)
                noisy_input.append(latents)

                _, x0_pred_cond = model(latents, conditional_dict, timestep)
                _, x0_pred_uncond = model(latents, unconditional_dict, timestep)

                x0_pred = x0_pred_uncond + args.guidance_scale * (x0_pred_cond - x0_pred_uncond)

                flow_pred = model._convert_x0_to_flow_pred(
                    scheduler=scheduler,
                    x0_pred=x0_pred.flatten(0, 1),
                    xt=latents.flatten(0, 1),
                    timestep=timestep.flatten(0, 1)
                ).unflatten(0, x0_pred.shape[:2])

                latents = scheduler.step(
                    flow_pred.flatten(0, 1),
                    scheduler.timesteps[progress_id] * torch.ones(
                        [1, num_frames], device=device, dtype=torch.long
                    ).flatten(0, 1),
                    latents.flatten(0, 1)
                ).unflatten(dim=0, sizes=flow_pred.shape[:2])

            noisy_input.append(latents)
            full_trajectory = torch.stack(noisy_input, dim=1)
            trajectory_sigmas = torch.cat([
                scheduler.sigmas,
                scheduler.sigmas.new_zeros(1),
            ])
            noisy_inputs = interpolate_trajectory(
                full_trajectory, trajectory_sigmas, aligned_sigmas
            )

            final_path = os.path.join(
                args.output_folder, f"{prompt_index:05d}.pt"
            )
            temporary_path = final_path + f".tmp.rank{rank}"
            torch.save(
                {prompt: noisy_inputs.cpu().detach()},
                temporary_path,
            )
            # Read back before publishing so resume never accepts a partial file.
            validation = torch.load(
                temporary_path, map_location="cpu", weights_only=True
            )
            saved_latent = next(iter(validation.values()))
            if (
                saved_latent.shape[1] != len(aligned_sigmas)
                or not torch.isfinite(saved_latent).all()
            ):
                raise RuntimeError("ODE pair readback validation failed")
            os.replace(temporary_path, final_path)
            processed += 1

        except Exception as e:
            logger.error(f"Failed prompt_index={prompt_index}: {e}")
            failed += 1
            continue

        if processed > 0 and processed % 50 == 0:
            elapsed = time.time() - start_time
            speed = processed / elapsed
            eta = (num_iters - index) / speed if speed > 0 else 0
            logger.info(
                f"Progress: processed={processed}, skipped={skipped}, failed={failed}, "
                f"speed={speed:.2f} samples/s, ETA={eta/60:.1f} min"
            )

    dist.barrier()
    elapsed = time.time() - start_time
    logger.info(
        f"Done! rank={rank}, processed={processed}, skipped={skipped}, "
        f"failed={failed}, elapsed={elapsed/60:.1f} min"
    )

    if rank == 0:
        final_count = len([f for f in os.listdir(args.output_folder) if f.endswith(".pt")])
        logger.info(f"Output directory: {final_count} .pt files")

    dist.barrier()


if __name__ == "__main__":
    main()
