#!/usr/bin/env bash
set -euo pipefail

export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export NCCL_TIMEOUT=3600  
export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export RAY_BACKEND_LOG_LEVEL=DEBUG



# verl path in src
REPO_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
SRC_DIR="$REPO_DIR/src"
export PYTHONPATH="$SRC_DIR${PYTHONPATH:+:$PYTHONPATH}"



PROJECT_NAME=Agentic-Verifier-GRPO  # project name
EXP_NAME=${EXP_NAME:-grpo-qwen3-4b-lora}

ACTOR_MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-4B}
SFT_ADAPTER_PATH=${SFT_ADAPTER_PATH:-null}

SAVE_BASE_DIR=${SAVE_BASE_DIR:-$REPO_DIR/checkpoints}
PROJECT_DIR=${PROJECT_NAME}/${EXP_NAME}

train_files=${TRAIN_FILES:-$REPO_DIR/data/grpo/train.parquet}
test_files=${VAL_FILES:-$REPO_DIR/data/grpo/test.parquet}

ROLLOUT_N=8

CONFIG_DIR=${SRC_DIR}/verl/config/RL
CONFIG_NAME=grpo_verifier # verl config name
REWAED_FN_PATH=${SRC_DIR}/verl/utils/reward_score/agent_verifier.py
AGENT_CONFIG_PATH=${AGENT_CONFIG_PATH:-$REPO_DIR/config/default.yml}
ENABLE_THINKING=True

n_gpus_per_node=${NGPUS_PER_NODE:-4}
nnodes=${NNODES:-1}

export WANDB_DIR=${SAVE_BASE_DIR}/${PROJECT_DIR}


NNODES=${nnodes}

NGPUS_PER_NODE=${n_gpus_per_node}

SHARED_ROOT=${SHARED_ROOT:-"/tmp"}

RAY_PORT=${RAY_PORT:-6380}

if (( NNODES == 1 )); then
    CLUSTER_DIR=$(mktemp -d "$SHARED_ROOT/agentv-grpo.XXXXXX")
else
    CLUSTER_DIR="$SHARED_ROOT/$EXP_NAME"
fi
LOCK_DIR="${CLUSTER_DIR}/lock"
HEAD_IP_FILE="${CLUSTER_DIR}/head_ip"

mkdir -p "${CLUSTER_DIR}"

THIS_IP=${THIS_IP:-$(hostname -I | awk '{print $1}')}

echo "[NODE] Hostname: $(hostname), IP: ${THIS_IP}"
echo "[NODE] Shared cluster dir: ${CLUSTER_DIR}"

if mkdir "${LOCK_DIR}" 2>/dev/null; then
    echo "[HEAD] I AM HEAD NODE."

    ray start --head \
        --node-ip-address="${THIS_IP}" \
        --port="${RAY_PORT}" \
        --num-gpus="${NGPUS_PER_NODE}"

    cleanup_head() {
        ray stop || true
        rm -f -- "$HEAD_IP_FILE"
        rmdir -- "$LOCK_DIR" "$CLUSTER_DIR" 2>/dev/null || true
    }
    trap cleanup_head EXIT

    echo "${THIS_IP}" > "${HEAD_IP_FILE}"
    echo "[HEAD] Ray head started at ${THIS_IP}:${RAY_PORT}"
    export RAY_ADDRESS="${THIS_IP}:${RAY_PORT}"
    echo "[HEAD] Export RAY_ADDRESS=${RAY_ADDRESS}"

    export DESIRED_GPUS=$((NNODES * NGPUS_PER_NODE))
    echo "[HEAD] Waiting for ${DESIRED_GPUS} GPUs in the cluster..."

    python - << 'PY'
import os, time, ray

addr = os.environ.get("RAY_ADDRESS", "auto")
desired = int(os.environ["DESIRED_GPUS"])

ray.init(address=addr)
for i in range(600):
    res = ray.cluster_resources()
    gpu = int(res.get("GPU", 0))
    print(f"[HEAD][WAIT] current GPUs = {gpu}, desired = {desired}")
    if gpu >= desired:
        print("[HEAD][WAIT] enough GPUs available, proceed.")
        break
    time.sleep(5)
else:
    print("[HEAD][WAIT] Timeout waiting for GPUs, proceed anyway (may crash in trainer).")
ray.shutdown()
PY
    echo "[HEAD] Launch training"
    python3 -m verl.trainer.main_ppo --config-path="$CONFIG_DIR" --config-name="$CONFIG_NAME" \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=128 \
    data.val_batch_size=512 \
    data.max_prompt_length=4200 \
    data.max_response_length=25600 \
    data.filter_overlong_prompts=True \
    data.truncation='right' \
    actor_rollout_ref.model.path="$ACTOR_MODEL_PATH" \
    actor_rollout_ref.model.lora_adapter_path="$SFT_ADAPTER_PATH" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.kl_loss_coef=0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=40960 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.max_model_len=32000 \
    actor_rollout_ref.rollout.per_round_max_tokens=4096 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.2 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.n=4 \
    actor_rollout_ref.rollout.forward_ratio=0.5 \
    actor_rollout_ref.extra.agent_config_path="$AGENT_CONFIG_PATH" \
    actor_rollout_ref.extra.use_multiturn_wrapper=True \
    actor_rollout_ref.extra.use_dynamic_sampling=True \
    actor_rollout_ref.extra.enable_thinking=${ENABLE_THINKING} \
    custom_reward_function.path=$REWAED_FN_PATH \
    custom_reward_function.name=compute_agentic_reward \
    algorithm.use_kl_in_reward=False \
    trainer.user_multi_stage=False \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXP_NAME \
    trainer.default_local_dir=${SAVE_BASE_DIR}/${PROJECT_DIR} \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.val_before_train=True \
    trainer.nnodes=$nnodes \
    trainer.save_freq=5 \
    trainer.test_freq=5 \
    trainer.rollout_data_dir=${SAVE_BASE_DIR}/${PROJECT_DIR}/rollout_data \
    trainer.validation_data_dir=${SAVE_BASE_DIR}/${PROJECT_DIR}/validation_data \
    trainer.verl_dir=$SRC_DIR \
    trainer.total_epochs=200 "$@"

    echo "[HEAD] Training finished. Stopping Ray..."
else
    echo "[WORKER] I AM WORKER NODE. Waiting for HEAD_IP..."

    until [ -f "${HEAD_IP_FILE}" ]; do
        sleep 3
    done

    HEAD_IP=$(cat "${HEAD_IP_FILE}")
    echo "[WORKER] Found HEAD_IP = ${HEAD_IP}"

    ray stop || true

    ray start \
        --address="${HEAD_IP}:${RAY_PORT}" \
        --num-gpus="${NGPUS_PER_NODE}"

    echo "[WORKER] Joined Ray cluster. Keeping alive..."
    tail -f /dev/null
fi




