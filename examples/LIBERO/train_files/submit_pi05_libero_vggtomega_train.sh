#!/bin/bash
set -eu

# ==============================================================================
# 1. 基础环境与全局变量
# ==============================================================================
NODE=$(hostname -s)
TIMESTAMP=$(date +%m%d-%H-%M)
CONDA_ENV="${CONDA_ENV:-lerobot}"
LEROBOT_ROOT="${LEROBOT_ROOT:-/mnt/nas/code/lerobot}"
TRAIN_SCRIPT_REL="examples/LIBERO/train_files/run_pi05_libero_vggtomega_train.sh"
TRAIN_SCRIPT="${LEROBOT_ROOT}/${TRAIN_SCRIPT_REL}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

export PYTHONNOUSERSITE=1
export TORCH_SHARED_MEMORY_STRATEGY=file_system
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export HF_HUB_DISABLE_SYMLINKS="${HF_HUB_DISABLE_SYMLINKS:-1}"

# ==============================================================================
# 2. 运行路径与输出配置 (Paths Config)
# OpenPI official: pi05_base -> pi05_libero, plus training-only frozen vggtomega scene alignment
# Scene reads only image (not action or other scene tokens); VLM/action cannot read scene. Inference stays plain PI0.5.
#   8 GPU, per-GPU batch 32 (global 256), 30000 steps, lr 5e-5
# ==============================================================================
export RUN_ID="${RUN_ID:-$(date +%m%d)_pi05_base_libero_vggtomega}"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-${LEROBOT_ROOT}/results/libero_PI}"
export OUTPUT_DIR="${OUTPUT_DIR:-${RUN_ROOT_DIR}/${RUN_ID}}"
export STEPS="${STEPS:-30000}"
export BATCH_SIZE="${BATCH_SIZE:-32}"
export SAVE_FREQ="${SAVE_FREQ:-${STEPS}}"
export NUM_WORKERS="${NUM_WORKERS:-4}"
export PI05_PRETRAINED="${PI05_PRETRAINED:-/mnt/nas/code/models/lerobot_pi05_base}"
export VGGTOMEGA_SOURCE_PATH="${VGGTOMEGA_SOURCE_PATH:-/mnt/nas/code/starVLA/third_party/vggtomega}"
export VGGTOMEGA_CHECKPOINT_PATH="${VGGTOMEGA_CHECKPOINT_PATH:-/mnt/nas/code/starVLA/third_party/vggtomega/vggt_omega_1b_512.pt}"
export VGGTOMEGA_ALIGN_LOSS_SCALE="${VGGTOMEGA_ALIGN_LOSS_SCALE:-0.05}"
export VGGTOMEGA_ALIGN_PROJECTOR_LR="${VGGTOMEGA_ALIGN_PROJECTOR_LR:-1e-4}"

# ==============================================================================
# 3. 分布式与硬件加速配置 (Distributed/NCCL Config)
# ==============================================================================
export NCCL_DEBUG=WARN
unset NCCL_IB_HCA
export NCCL_IB_DISABLE=1
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=10000
export NCCL_SOCKET_TIMEOUT_MS=360000
export NCCL_SOCKET_FAMILY=AF_INET
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29600}"

if ip -o link show eth0 >/dev/null 2>&1; then
  export NCCL_SOCKET_IFNAME=eth0
elif ip -o link show net0 >/dev/null 2>&1; then
  export NCCL_SOCKET_IFNAME=net0
else
  unset NCCL_SOCKET_IFNAME
fi
export GLOO_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-}"

if command -v nvidia-smi >/dev/null 2>&1; then
  AVAILABLE_GPUS="$(nvidia-smi -L | wc -l | tr -d ' ')"
else
  AVAILABLE_GPUS=1
fi
OFFICIAL_NUM_GPUS=8
export NUM_PROCESSES="${NUM_PROCESSES:-${OFFICIAL_NUM_GPUS}}"
if [ "${NUM_PROCESSES}" -gt "${AVAILABLE_GPUS}" ]; then
  echo "WARN: official recipe is ${OFFICIAL_NUM_GPUS} GPUs, but only ${AVAILABLE_GPUS} are visible; using ${AVAILABLE_GPUS}."
  export NUM_PROCESSES="${AVAILABLE_GPUS}"
fi

export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_ENABLE="${WANDB_ENABLE:-true}"

# ── 工具函数 ──────────────────────────────────────────────────────
log()     { printf '[%s] [%s] %s\n' "$(date '+%H:%M:%S')" "$NODE" "$*"; }
hr()      { printf -- '--------------------------------------------------------\n'; }
ok()      { log "OK $*"; }
die()     { log "ERR $*" >&2; exit 1; }
elapsed() { printf '%ds' "$(( $(date +%s) - $1 ))"; }

# ── 检查与准备 ────────────────────────────────────────────────────
hr
log "=== π₀.₅ base -> libero + vggtomega scene tokens ==="
log "host       : $(hostname -f)"
log "start time : $(date '+%Y-%m-%d %H:%M:%S')"
log "timestamp  : ${TIMESTAMP}"
log "repo root  : $LEROBOT_ROOT"
log "script     : $TRAIN_SCRIPT_REL"
hr

[ -d "$LEROBOT_ROOT" ] || die "lerobot not found: $LEROBOT_ROOT"
[ -f "$TRAIN_SCRIPT" ] || die "train script not found: $TRAIN_SCRIPT"

# ── ① 激活环境 ─────────────────────────────────────────────────────
hr
log "Activating conda env: $CONDA_ENV"
T0=$(date +%s)

if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  . "$(conda info --base)/etc/profile.d/conda.sh"
elif [ -f "/root/miniconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  . /root/miniconda3/etc/profile.d/conda.sh
else
  die "conda not found, please load conda first"
fi
conda activate "$CONDA_ENV"

ok "Conda ready  (${CONDA_DEFAULT_ENV:-unknown}, $(elapsed $T0))"

# W&B uses existing credentials or an explicitly supplied environment key.
if [ "${WANDB_ENABLE}" = "true" ] && [ "${WANDB_MODE}" != "offline" ] && [ -n "${WANDB_API_KEY:-}" ]; then
  command -v wandb >/dev/null 2>&1 || die "wandb command not found in env: $CONDA_ENV"
  wandb login "$WANDB_API_KEY" >/dev/null
fi

# ── ② 打印训练配置 ────────────────────────────────────────────────
hr
log "train config :"
log "  LEROBOT_ROOT      : $LEROBOT_ROOT"
log "  TRAIN_SCRIPT      : $TRAIN_SCRIPT_REL"
log "  RUN_ROOT_DIR      : $RUN_ROOT_DIR"
log "  RUN_ID            : $RUN_ID"
log "  OUTPUT_DIR        : $OUTPUT_DIR"
log "  PI05_PRETRAINED   : $PI05_PRETRAINED"
log "  VGGTOMEGA_SOURCE_PATH: $VGGTOMEGA_SOURCE_PATH"
log "  VGGTOMEGA_CHECKPOINT_PATH: $VGGTOMEGA_CHECKPOINT_PATH"
log "  VGGTOMEGA_ALIGN_LOSS_SCALE: $VGGTOMEGA_ALIGN_LOSS_SCALE"
log "  VGGTOMEGA_ALIGN_PROJECTOR_LR: $VGGTOMEGA_ALIGN_PROJECTOR_LR"
log "  NUM_PROCESSES     : $NUM_PROCESSES  (available=${AVAILABLE_GPUS}, OpenPI=8)"
log "  BATCH_SIZE        : $BATCH_SIZE per GPU  (effective=$((BATCH_SIZE * NUM_PROCESSES)), OpenPI=256)"
log "  STEPS             : $STEPS  (OpenPI=30000)"
log "  SAVE_FREQ         : $SAVE_FREQ"
log "  MASTER_ADDR:PORT  : ${MASTER_ADDR}:${MASTER_PORT}"
log "  NCCL_SOCKET_IFNAME: ${NCCL_SOCKET_IFNAME:-unset}"
log "  WANDB_MODE        : $WANDB_MODE"
hr

# ── ③ 启动训练 ────────────────────────────────────────────────────
hr
TRAIN_START=$(date +%s)
log "Starting train -> bash $TRAIN_SCRIPT_REL"

cd "$LEROBOT_ROOT"
bash "$TRAIN_SCRIPT_REL"

hr
ok "Train finished  (total wall time: $(elapsed $TRAIN_START))"
hr
