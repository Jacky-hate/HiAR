#!/bin/bash
# Generate ODE trajectory pairs for forward-KL regularization.
#
# Multi-node example:
#   Node 0: NODE_RANK=0 MASTER_ADDR=<ip> bash scripts/run_generate_ode_pairs.sh
#   Node 1: NODE_RANK=1 MASTER_ADDR=<ip> bash scripts/run_generate_ode_pairs.sh
#
# Single-node example:
#   NNODES=1 NPROC_PER_NODE=8 bash scripts/run_generate_ode_pairs.sh
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"

# ---- Cluster configuration ----
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-29500}

# ---- Sampling parameters ----
CAPTION_PATH=${CAPTION_PATH:-"data/vidprom_filtered_extended.txt"}
OUTPUT_FOLDER=${OUTPUT_FOLDER:-"data/ode_pairs"}
GUIDANCE_SCALE=${GUIDANCE_SCALE:-6.0}
NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS:-50}
NUM_PROMPTS=${NUM_PROMPTS:-20000}
STUDENT_DENOISING_STEPS=${STUDENT_DENOISING_STEPS:-"1000,750,500,250,0"}
STUDENT_TIMESTEP_SHIFT=${STUDENT_TIMESTEP_SHIFT:-5.0}
LATENT_SHAPE=${LATENT_SHAPE:-"1,21,16,60,104"}
NUM_FRAMES=${NUM_FRAMES:-21}
MODEL_NAME=${MODEL_NAME:-"Wan2.1-T2V-1.3B"}
TIMESTEP_SHIFT=${TIMESTEP_SHIFT:-8.0}
SEED=${SEED:-42}
RESUME_FLAG=${RESUME_FLAG:-"--resume"}

echo "========================================"
echo "ODE Trajectory Pair Generation"
echo "========================================"
echo "Nodes: ${NNODES}, GPUs/node: ${NPROC_PER_NODE}, Total GPUs: $((NNODES * NPROC_PER_NODE))"
echo "Prompts: ${CAPTION_PATH} (using ${NUM_PROMPTS})"
echo "Output:  ${OUTPUT_FOLDER}"
echo "========================================"

torchrun \
    --nnodes=${NNODES} \
    --nproc_per_node=${NPROC_PER_NODE} \
    --node_rank=${NODE_RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    scripts/generate_ode_pairs.py \
    --output_folder ${OUTPUT_FOLDER} \
    --caption_path ${CAPTION_PATH} \
    --guidance_scale ${GUIDANCE_SCALE} \
    --num_inference_steps ${NUM_INFERENCE_STEPS} \
    --num_prompts ${NUM_PROMPTS} \
    --student_denoising_steps ${STUDENT_DENOISING_STEPS} \
    --student_timestep_shift ${STUDENT_TIMESTEP_SHIFT} \
    --latent_shape ${LATENT_SHAPE} \
    --num_frames ${NUM_FRAMES} \
    --model_name ${MODEL_NAME} \
    --timestep_shift ${TIMESTEP_SHIFT} \
    --seed ${SEED} \
    ${RESUME_FLAG}

echo "Done! Output: ${OUTPUT_FOLDER}"
