#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

NPROC_PER_NODE=${NPROC_PER_NODE:-4}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-4B}
TRAIN_FILES=${TRAIN_FILES:-$REPO_DIR/data/sft/sft_multiturn_train.parquet}
VAL_FILES=${VAL_FILES:-$REPO_DIR/data/sft/sft_multiturn_test.parquet}
SFT_OUTPUT_DIR=${SFT_OUTPUT_DIR:-$REPO_DIR/checkpoints/sft-qwen3-4b-lora}

torchrun --standalone --nnodes=1 --nproc-per-node="$NPROC_PER_NODE" \
    -m verl.trainer.fsdp_sft_trainer \
    --config-path "$REPO_DIR/src/verl/config/SFT" \
    --config-name multiturn_sft_trainer \
    data.train_files="$TRAIN_FILES" \
    data.val_files="$VAL_FILES" \
    data.train_batch_size=8 \
    data.micro_batch_size_per_gpu=1 \
    data.max_length=21600 \
    model.partial_pretrain="$MODEL_PATH" \
    trainer.n_gpus_per_node="$NPROC_PER_NODE" \
    trainer.save_freq=65 \
    trainer.test_freq=500 \
    trainer.default_local_dir="$SFT_OUTPUT_DIR" \
    "$@"
