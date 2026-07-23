#!/usr/bin/env python3
"""Export a sharded DCP EMA checkpoint as a standalone inference checkpoint."""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from omegaconf import OmegaConf

from utils.distributed import FSDPModelState
from utils.wan_wrapper import WanDiffusionWrapper


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    ema_dir = checkpoint_dir / "ema_state"
    success_path = checkpoint_dir / "_SUCCESS"
    metadata_path = checkpoint_dir / "metadata.json"
    output_path = Path(args.output).resolve()

    if not success_path.is_file():
        raise FileNotFoundError(f"checkpoint is not committed: {success_path}")
    if not ema_dir.is_dir():
        raise FileNotFoundError(f"EMA state directory is missing: {ema_dir}")
    metadata = {}
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text())

    config = OmegaConf.merge(
        OmegaConf.load("configs/default_config.yaml"),
        OmegaConf.load(args.config_path),
    )
    model_kwargs = OmegaConf.to_container(
        config.get("model_kwargs", {}), resolve=True
    )
    print(f"Loading base generator on CPU with kwargs={model_kwargs}", flush=True)
    generator = WanDiffusionWrapper(**model_kwargs, is_causal=True)
    generator.eval().requires_grad_(False)

    print(f"Resharding EMA from {ema_dir}", flush=True)
    dcp.load(
        {"ema": FSDPModelState(generator)},
        checkpoint_id=str(ema_dir),
    )

    state_dict = generator.state_dict()
    meta_keys = [key for key, value in state_dict.items() if value.is_meta]
    if meta_keys:
        raise RuntimeError(f"EMA export has meta tensors: {meta_keys[:5]}")
    nonfinite = []
    for key, value in state_dict.items():
        if value.is_floating_point() and not torch.isfinite(value).all():
            nonfinite.append(key)
    if nonfinite:
        raise RuntimeError(f"EMA export has non-finite tensors: {nonfinite[:5]}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(
        f".{output_path.name}.tmp.{os.getpid()}"
    )
    payload = {
        "generator_ema": state_dict,
        "step": int(metadata.get("step", -1)),
        "source_checkpoint": str(checkpoint_dir),
        "checkpoint_metadata": metadata,
    }
    try:
        torch.save(payload, temp_path)
        os.replace(temp_path, output_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()

    print(
        f"EXPORT_COMPLETE output={output_path} step={payload['step']} "
        f"tensors={len(state_dict)} bytes={output_path.stat().st_size}",
        flush=True,
    )


if __name__ == "__main__":
    main()
