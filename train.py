import argparse
import os
from omegaconf import OmegaConf
try:
    import wandb
except ModuleNotFoundError:
    wandb = None

from trainer import ScoreDistillationTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--no_visualize", action="store_true")
    parser.add_argument("--logdir", type=str, default="", help="Path to the directory to save logs")
    parser.add_argument("--wandb-save-dir", type=str, default="", help="Path to the directory to save wandb logs")
    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument("--resume", type=str, default="",
                        help="Resume training from checkpoint. Use 'auto' to auto-detect latest checkpoint, "
                             "or specify the path to model.pt file")
    parser.add_argument("--min_exit_step", type=int, default=None,
                        help="Override min_exit_step in config. Controls the minimum exit step index "
                             "(i.e. randint low), e.g. 2 means skip step 0 and 1")
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Stop when the absolute global outer-step reaches this value",
    )
    parser.add_argument(
        "--total_batch_size",
        type=int,
        default=None,
        help="Override the global batch size used to derive accumulation",
    )
    parser.add_argument(
        "--num_rollout_frames",
        type=int,
        default=None,
        help="Override rollout length for long-rollout validation/training",
    )
    parser.add_argument(
        "--fixed_rollout_frames",
        type=int,
        default=None,
        help="Force an exact rollout length instead of uniform length sampling",
    )

    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)
    default_config = OmegaConf.load("configs/default_config.yaml")
    config = OmegaConf.merge(default_config, config)
    config.no_save = args.no_save
    config.no_visualize = args.no_visualize

    config_name = os.path.basename(args.config_path).split(".")[0]
    config.config_name = config_name
    config.logdir = args.logdir
    config.wandb_save_dir = args.wandb_save_dir
    config.disable_wandb = args.disable_wandb

    if args.resume:
        config.resume_from = args.resume

    if args.min_exit_step is not None:
        config.min_exit_step = args.min_exit_step
    if args.max_steps is not None:
        config.max_steps = args.max_steps
    if args.total_batch_size is not None:
        config.total_batch_size = args.total_batch_size
    if args.num_rollout_frames is not None:
        config.num_rollout_frames = args.num_rollout_frames
    if args.fixed_rollout_frames is not None:
        config.fixed_rollout_frames = args.fixed_rollout_frames

    trainer = ScoreDistillationTrainer(config)
    trainer.train()

    if wandb is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
