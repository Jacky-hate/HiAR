<p align="center">
<h1 align="center">HiAR</h1>
<h3 align="center">Hierarchical Autoregressive Video Generation with Pipelined Parallel Inference</h3>
</p>
<p align="center">
  <h3 align="center"><a href="https://arxiv.org/abs/2603.08703">arXiv</a> | <a href="https://jacky-hate.github.io/HiAR/">Website</a> | <a href="https://huggingface.co/jackyhate/HiAR/tree/main">Model</a></h3>
</p>

---

HiAR proposes **hierarchical denoising** for autoregressive video diffusion models, a paradigm shift from conventional block-first to **step-first** denoising order. By conditioning each block on context at a matched noise level, HiAR maximally attenuates error propagation while preserving temporal causality, achieving **state-of-the-art long video generation** (20s+) with significantly reduced quality drift.

---

## Installation
Create a conda environment and install dependencies:
```
conda create -n hiar python=3.10 -y
conda activate hiar
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

## Quick Start
### Download checkpoints
```bash
# Download Wan2.1 base model
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir-use-symlinks False --local-dir wan_models/Wan2.1-T2V-1.3B

# Download HiAR checkpoint
mkdir -p ckpts
wget -O ckpts/hiar.pt https://huggingface.co/jackyhate/HiAR/resolve/main/hiar.pt
```

### Inference
Generate short videos (~5 seconds, 21 latent frames):
```bash
python inference.py \
    --config_path configs/hiar.yaml \
    --checkpoint_path ckpts/hiar.pt \
    --data_path data/prompts.txt \
    --output_folder outputs/ \
    --num_output_frames 21 \
    --use_ema \
    --inference_method timestep_first
```

Generate long videos (~20 seconds, 81 latent frames):
```bash
python inference.py \
    --config_path configs/hiar.yaml \
    --checkpoint_path ckpts/hiar.pt \
    --data_path data/prompts.txt \
    --output_folder outputs/ \
    --num_output_frames 81 \
    --use_ema \
    --inference_method timestep_first
```


### Pipelined Parallel Inference

HiAR's hierarchical denoising structure naturally admits **pipelined parallel inference**: each GPU handles one denoising stage, and blocks flow through a diagonal pipeline schedule. 

```bash
# Requires exactly N GPUs for N denoising steps (default: 4)
torchrun --nproc_per_node=4 \
    scripts/pipeline_parallel_inference.py \
    --config_path configs/hiar.yaml \
    --checkpoint_path ckpts/hiar.pt \
    --prompt "A cat sitting on a windowsill watching the rain fall outside" \
    --output_path outputs/pipeline_output.mp4 \
    --num_output_frames 81
```



## Training

Our training algorithm is data-free (**no video data is needed**), requiring only text prompts and pre-generated ODE trajectory pairs from the teacher model for forward-KL regularization.

### Step 1: Download training prerequisites
```bash
# Download Wan2.1 teacher model (14B, for real score computation)
huggingface-cli download Wan-AI/Wan2.1-T2V-14B --local-dir-use-symlinks False --local-dir wan_models/Wan2.1-T2V-14B

# Download ODE initialization checkpoint from the official Self-Forcing release
mkdir -p ckpts
wget -O ckpts/init.pt https://huggingface.co/gdhe17/Self-Forcing/resolve/main/checkpoints/ode_init.pt
```

### Step 2: Prepare training prompts

We use the same VidProM-filtered prompt set as [Self-Forcing](https://github.com/self-forcing/Self-Forcing). Download the prompt file from the Self-Forcing release:
```bash
mkdir -p data
wget -O data/vidprom_filtered_extended.txt https://huggingface.co/gdhe17/Self-Forcing/resolve/main/prompts/vidprom_filtered_extended.txt
```

### Step 3: Generate ODE trajectory pairs (for forward-KL)

Pre-generate ODE trajectory pairs from the teacher model. These serve as targets for the forward-KL regularizer that prevents low-motion shortcuts:
```bash
# Single-node, 8 GPUs (generates ~16K pairs by default)
bash scripts/run_generate_ode_pairs.sh

# Multi-node example
NNODES=4 NPROC_PER_NODE=8 NODE_RANK=0 MASTER_ADDR=<ip> \
    bash scripts/run_generate_ode_pairs.sh
```

The script supports resuming from interruptions with `--resume`. Adjust `NUM_PROMPTS` to control how many pairs to generate (10K-50K recommended).

### Step 4: Run HiAR training
```bash
torchrun --nnodes=8 --nproc_per_node=8 --rdzv_id=5235 \
    --rdzv_backend=c10d \
    --rdzv_endpoint $MASTER_ADDR \
    train.py \
    --config_path configs/hiar.yaml \
    --logdir logs/hiar \
    --disable-wandb
```


---
## Discussion & Limitations
In essence, this method allows autoregressive video generation to mimic a bidirectional attention video denoising paradigm. For instance, the high-noise denoising stages only require coarse-grained context information. This design maximally reduces error accumulation while theoretically retaining sufficient information to maintain continuity. By scaling the training budget under the constraint of the Forward KL loss, we can achieve near-zero degradation in most scenarios, even enabling infinite generation (e.g., over 200 minutes). However, in some dynamic scenes, inter-frame jumping may still occur. We believe this is not an inherent limitation of the hierarchical denoising mechanism itself, but rather an issue of insufficient capacity in the 1.3B base model, as this denoising paradigm is considerably more challenging. We plan to further validate this mechanism on more powerful base models in the future.

## Acknowledgements
This codebase is built on [Self-Forcing](https://github.com/self-forcing/Self-Forcing), [CausVid](https://github.com/tianweiy/CausVid), and [Wan2.1](https://github.com/Wan-Video/Wan2.1).

## Citation
```bibtex
@misc{zou2026hiarefficientautoregressivelong,
      title={HiAR: Efficient Autoregressive Long Video Generation via Hierarchical Denoising}, 
      author={Kai Zou and Dian Zheng and Hongbo Liu and Tiankai Hang and Bin Liu and Nenghai Yu},
      year={2026},
      eprint={2603.08703},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2603.08703}, 
}
```
