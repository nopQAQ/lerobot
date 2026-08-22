#!/usr/bin/env bash
# Install LeRobot + PPU PyTorch on Aliyun PPU (PG1) nodes.
# Reference: https://help.aliyun.com/zh/document_detail/2997023.html
#
# IMPORTANT:
# - Do NOT use HTTP proxy when installing from PPU pip index (causes SSL failures).
# - PPU v1.7.x unified index (OS/CUDA free):
#   https://aiext-pypi.mirrors.aliyuncs.com/pg1-pip/pypi_index/simple/
# - On Ubuntu 22.04 + CUDA 12.4, PPU currently ships torch 2.6.0 for Python 3.10.
#   LeRobot 0.6.1 requires Python >=3.12 and torch >=2.7; upgrade when PTG releases
#   cp312+cuda124 wheels, or use Ubuntu 24.04 base image.

set -euo pipefail

ENV_NAME="${ENV_NAME:-lerobot}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PPU_INDEX="${PPU_INDEX:-https://aiext-pypi.mirrors.aliyuncs.com/pg1-pip/pypi_index/simple/}"
PYPI_INDEX="${PYPI_INDEX:-https://pypi.org/simple}"

echo "[1/6] Create conda env: ${ENV_NAME} (python=${PYTHON_VERSION})"
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "  env ${ENV_NAME} already exists, reusing"
else
  conda create -n "${ENV_NAME}" "python=${PYTHON_VERSION}" -y
fi

CONDA_BASE="$(conda info --base)"
ENV_PREFIX="${CONDA_BASE}/envs/${ENV_NAME}"
PIP="${ENV_PREFIX}/bin/pip"
PY="${ENV_PREFIX}/bin/python"
if [[ ! -x "${PIP}" ]]; then
  echo "ERROR: pip not found at ${PIP}" >&2
  exit 1
fi

echo "[2/6] Install PPU torch/torchvision (no proxy)"
# Unset proxy + PIP_INDEX_URL so PPU selector works correctly.
env -u PIP_INDEX_URL -u PPU_INDEX \
    -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u all_proxy \
    "${PIP}" install "torch==2.6.0" \
    -i "${PPU_INDEX}" \
    --trusted-host aiext-pypi.mirrors.aliyuncs.com \
    --timeout 600

# torchvision often needs torch pre-installed; use matching PPU version when available.
if ! env -u PIP_INDEX_URL -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u all_proxy \
    "${PIP}" install "torchvision==0.21.0" \
    -i "${PPU_INDEX}" \
    --trusted-host aiext-pypi.mirrors.aliyuncs.com \
    --timeout 600 2>/dev/null; then
  echo "  torchvision from PPU index failed; trying cached/local wheel if present"
  TV_WHEEL="$(find /root/.cache/pip -name 'torchvision-0.21.0+ppu*.whl' 2>/dev/null | head -1 || true)"
  if [[ -n "${TV_WHEEL}" ]]; then
    env -u PIP_INDEX_URL -u PPU_INDEX "${PIP}" install "${TV_WHEEL}" -i "${PYPI_INDEX}"
  else
    echo "ERROR: torchvision wheel not found" >&2
    exit 1
  fi
fi

echo "[3/6] Verify PPU torch"
cd /tmp
"${PY}" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

echo "[4/6] Pin base deps"
env -u PIP_INDEX_URL -u PPU_INDEX "${PIP}" install \
  "setuptools>=71.0.0,<81.0.0" \
  "numpy>=2.0.0,<2.3.0" \
  -i "${PYPI_INDEX}"

echo "[5/6] Install LeRobot (pi + training + libero extras)"
cd "${REPO_ROOT}"
env -u PIP_INDEX_URL -u PPU_INDEX "${PIP}" install -e ".[pi,training,libero]" \
  --no-deps --ignore-requires-python \
  -i "${PYPI_INDEX}"

env -u PIP_INDEX_URL -u PPU_INDEX "${PIP}" install \
  "draccus==0.10.0" \
  "huggingface-hub>=1.0.0,<2.0.0" \
  "requests>=2.32.0,<3.0.0" \
  "gymnasium>=1.1.1,<2.0.0" \
  "safetensors>=0.4.3,<1.0.0" \
  "packaging>=24.2,<26.0" \
  "termcolor>=2.4.0,<4.0.0" \
  "tqdm>=4.66.0,<5.0.0" \
  "cmake>=3.29.0.1,<4.2.0" \
  "opencv-python-headless>=4.9.0,<4.14.0" \
  "einops>=0.8.0,<0.9.0" \
  "datasets>=4.7.0,<5.0.0" \
  "pandas>=2.0.0,<3.0.0" \
  "pyarrow>=21.0.0,<30.0.0" \
  "av>=15.0.0,<16.0.0" \
  "jsonlines>=4.0.0,<5.0.0" \
  "wandb>=0.24.0,<0.28.0" \
  "accelerate>=1.14.0,<2.0.0" \
  "transformers>=5.4.0,<5.6.0" \
  "scipy>=1.14.0,<2.0.0" \
  "hf-libero>=0.1.4,<0.2.0" \
  -i "${PYPI_INDEX}"

echo "[6/7] Optional: typing.Self shim for Python <3.11"
SITE_PACKAGES="$("${PY}" -c 'import site; print(site.getsitepackages()[0])')"
cat > "${SITE_PACKAGES}/sitecustomize.py" <<'PYEOF'
import typing
try:
    from typing import Self  # noqa: F401
except ImportError:
    from typing_extensions import Self
    typing.Self = Self
PYEOF
env -u PIP_INDEX_URL -u PPU_INDEX "${PIP}" install typing_extensions -i "${PYPI_INDEX}" -q

echo "[7/7] Done"
echo
echo "Activate: conda activate ${ENV_NAME}"
echo
echo "NOTE: LeRobot 0.6.1 officially requires Python >=3.12 and torch >=2.7."
echo "This node (Ubuntu 22.04 + CUDA 12.4) currently has PPU torch 2.6.0 for Python 3.10 only."
echo "When PTG releases cp312+cuda124 wheels, recreate env with:"
echo "  PYTHON_VERSION=3.12 bash $0"
echo "and install torch==2.7.0 torchvision==0.22.0 from PPU index."
