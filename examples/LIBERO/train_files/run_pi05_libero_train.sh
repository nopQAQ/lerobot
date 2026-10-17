#!/usr/bin/env bash
# OpenPI official: pi05_base -> pi05_libero
# Source: Physical-Intelligence/openpi src/openpi/training/config.py  name="pi05_libero"
set -euo pipefail

export PYTHONNOUSERSITE=1
export TORCH_SHARED_MEMORY_STRATEGY=file_system
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export HF_HUB_DISABLE_SYMLINKS="${HF_HUB_DISABLE_SYMLINKS:-1}"
unset NCCL_IB_HCA
export NCCL_IB_DISABLE=1

if ip -o link show eth0 >/dev/null 2>&1; then
  export NCCL_SOCKET_IFNAME=eth0
elif ip -o link show net0 >/dev/null 2>&1; then
  export NCCL_SOCKET_IFNAME=net0
else
  unset NCCL_SOCKET_IFNAME
fi

export GLOO_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-}"
export NCCL_SOCKET_FAMILY=AF_INET
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29600}"
export NCCL_DEBUG="${NCCL_DEBUG:-NONE}"
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=10000
export NCCL_SOCKET_TIMEOUT_MS=360000

###########################################################################################
# OpenPI TrainConfig(name="pi05_libero"):
#   weight_loader = pi05_base
#   data          = physical-intelligence/libero  (LeRobot copy: HuggingFaceVLA/libero)
#   batch_size    = 256
#   num_train_steps = 30_000
#   peak_lr       = 5e-5, warmup=10_000, decay_lr=5e-5
#   action_horizon = 10, extra_delta_transform=False
#   AdamW clip=1.0, weight_decay=1e-10
# === Please modify the following paths according to your environment ===
config_yaml=./examples/LIBERO/train_files/pi05_libero.yaml
run_id="${RUN_ID:-$(date +%m%d)_pi05_base_libero}"
# === End of environment variable configuration ===
###########################################################################################

export WANDB_MODE="${WANDB_MODE:-online}"

project_root="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "${project_root}"

libero_root="${LIBERO_DATA_ROOT:-${project_root}/playground/Datasets/HuggingFaceVLA/libero}"
output_dir="${OUTPUT_DIR:-${project_root}/results/libero_PI/${run_id}}"

resolve_pretrained() {
  if [ -n "${PI05_PRETRAINED:-}" ]; then
    printf '%s\n' "${PI05_PRETRAINED}"
    return
  fi
  local candidates=(
    "/mnt/nas/code/models/lerobot_pi05_base"
    "${project_root}/playground/Pretrained_models/pi05_base"
  )
  local p
  for p in "${candidates[@]}"; do
    if [ -f "${p}/model.safetensors" ] || [ -f "${p}/config.json" ]; then
      printf '%s\n' "${p}"
      return
    fi
  done
  echo "ERROR: pi05_base not found. Set PI05_PRETRAINED or place weights at /mnt/nas/code/models/lerobot_pi05_base." >&2
  exit 1
}

pretrained="$(resolve_pretrained)"

if [ ! -f "${libero_root}/meta/info.json" ]; then
  echo "ERROR: LIBERO dataset not found: ${libero_root}" >&2
  echo "Hint: expected playground/Datasets/HuggingFaceVLA/libero (OpenPI: physical-intelligence/libero)." >&2
  exit 1
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  available_gpus="$(nvidia-smi -L | wc -l | tr -d ' ')"
else
  available_gpus="$(python -c 'import torch; print(max(torch.cuda.device_count(), 1))')"
fi

# OpenPI global batch 256 on 8 GPUs -> 32 / GPU. Fall back to visible count.
official_num_gpus=8
num_processes="${NUM_PROCESSES:-${official_num_gpus}}"
if [ "${num_processes}" -gt "${available_gpus}" ]; then
  echo "WARN: OpenPI uses ${official_num_gpus} GPUs (global batch 256), but only ${available_gpus} are visible; using ${available_gpus}."
  num_processes="${available_gpus}"
fi

DEBUG="${DEBUG:-false}"
if [ "${DEBUG}" = "true" ]; then
  num_processes=1
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
fi

steps="${STEPS:-30000}"
batch_size="${BATCH_SIZE:-32}"
# Default: only keep the final checkpoint (LeRobot also always saves at step==steps).
save_freq="${SAVE_FREQ:-${steps}}"
wandb_enable="${WANDB_ENABLE:-true}"
num_workers="${NUM_WORKERS:-4}"
effective_batch="$((batch_size * num_processes))"

# pi05_base uses OpenPI camera names. Map LIBERO image/image2 onto them.
# Do not set empty_cameras: base already has 3 cameras; missing wrist is zero-filled.
extra_args=()
if grep -q 'base_0_rgb' "${pretrained}/config.json" 2>/dev/null; then
  extra_args+=(
    --rename_map='{"observation.images.image":"observation.images.base_0_rgb","observation.images.image2":"observation.images.left_wrist_0_rgb"}'
  )
fi

accelerate_args=(--mixed_precision=bf16 --num_processes="${num_processes}")
if [ "${num_processes}" -gt 1 ]; then
  accelerate_args+=(--multi_gpu)
fi

echo "project_root: ${project_root}"
echo "config_yaml: ${config_yaml}"
echo "run_id: ${run_id}"
echo "pretrained: ${pretrained}  (OpenPI start: pi05_base)"
echo "libero_root: ${libero_root}"
echo "output_dir: ${output_dir}"
echo "num_processes: ${num_processes}  (available=${available_gpus}, OpenPI=8)"
echo "batch_size: ${batch_size} per GPU  (effective=${effective_batch}, OpenPI=256)"
echo "steps: ${steps}  (OpenPI=30000)"
echo "lr: 5e-5  warmup=10000  decay_lr=5e-5"
echo "save_freq: ${save_freq}"
echo "DEBUG: ${DEBUG}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-}"
echo "WANDB_ENABLE: ${wandb_enable}"

# Do not mkdir output_dir: lerobot-train refuses to start if it already exists.

# Scheduler: OpenPI decay_steps=1e6 would auto-scale warmup in LeRobot
# (steps=30k < 1e6). Set decay_steps=steps and decay_lr=peak so LR stays 5e-5
# after the 10k warmup — same curve as OpenPI over 30k steps.
# V-JEPA prefix scene tokens: use run_pi05_libero_vjepa_train.sh
accelerate launch \
  "${accelerate_args[@]}" \
  "$(command -v lerobot-train)" \
  --policy.path="${pretrained}" \
  --policy.push_to_hub=false \
  --policy.dtype=bfloat16 \
  --policy.gradient_checkpointing=true \
  --policy.compile_model=false \
  --policy.freeze_vision_encoder=false \
  --policy.train_expert_only=false \
  --policy.device=cuda \
  --policy.chunk_size=10 \
  --policy.n_action_steps=10 \
  --policy.use_relative_actions=false \
  --policy.normalization_mapping='{"ACTION":"MEAN_STD","STATE":"MEAN_STD","VISUAL":"IDENTITY"}' \
  --policy.optimizer_lr=5e-5 \
  --policy.optimizer_weight_decay=1e-10 \
  --policy.optimizer_grad_clip_norm=1.0 \
  --policy.scheduler_warmup_steps=10000 \
  --policy.scheduler_decay_steps="${steps}" \
  --policy.scheduler_decay_lr=5e-5 \
  --dataset.repo_id=HuggingFaceVLA/libero \
  --dataset.root="${libero_root}" \
  --output_dir="${output_dir}" \
  --job_name="${run_id}" \
  --seed=42 \
  --batch_size="${batch_size}" \
  --steps="${steps}" \
  --save_freq="${save_freq}" \
  --log_freq=100 \
  --num_workers="${num_workers}" \
  --wandb.enable="${wandb_enable}" \
  --wandb.project=lerobot_liberoPI \
  "${extra_args[@]}"
