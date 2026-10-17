# LIBERO π₀.₅ training files

Mapped from `starVLA/examples/LIBERO/train_files`.

| starVLA | this folder |
|---------|-------------|
| `starvla_cotrain_liberoPI.yaml` | `pi05_libero.yaml` |
| `run_liberoPI_Qwen2B_train.sh` | `run_pi05_libero_train.sh` |
| `run_liberoOFT_Qwen2B_resume.sh` | `run_pi05_libero_resume.sh` |
| DSW/HPC train bootstrap | `submit_pi05_libero_train.sh` |
| `playground/Datasets/LEROBOT_LIBERO_DATA` | `playground/Datasets/HuggingFaceVLA/libero` |
| `results/liberoPI` | `results/libero_PI` |

V-JEPA scene-token recipe (same OpenPI hyperparameters, alignment on; scene tokens sit after the action suffix):

| file | role |
|------|------|
| `pi05_libero_vjepa.yaml` | hyperparameter sheet (`vjepa_enabled=true`, 8-frame clip) |
| `run_pi05_libero_vjepa_train.sh` | train launch |
| `run_pi05_libero_vjepa_resume.sh` | resume from `checkpoints/last` |
| `submit_pi05_libero_vjepa_train.sh` | DSW/HPC bootstrap |

LeRobot does not load the YAML at runtime. Edit the YAML as the hyperparameter sheet, and keep the matching launch script in sync.

Checkpoints go to `results/libero_PI/<run_id>/`.

## Official recipe

From [LeRobot LIBERO docs](https://huggingface.co/docs/lerobot/libero): finetune `pi05_libero` for 6000 steps in bf16, **batch 256 on 8 H100** (`32 × 8`).

| item | official | this node |
|------|----------|-----------|
| GPUs | 8×H100 | default 8, auto-fall back to visible count (here 4) |
| per-GPU batch | 32 | 32 (`BATCH_SIZE=8` if OOM) |
| effective batch | 256 | 128 on 4 GPUs |
| steps | 6000 | 6000 |
| dtype | bfloat16 | bfloat16 |
| compile | true | **false** (PPU) |
| wandb | enable | enable (`WANDB_MODE=online`) |

This folder's default `STEPS` is **30000**, matching OpenPI `pi05_libero`.
Default `SAVE_FREQ` equals `STEPS`, so only the final 30k checkpoint is kept (`SAVE_FREQ=5000` restores intermediates).

## Train (baseline π₀.₅)

DSW / job bootstrap (activates conda, optional W&B login, then launches training):

```bash
export WANDB_API_KEY=...          # required when WANDB_MODE=online
bash /mnt/nas/code/lerobot/examples/LIBERO/train_files/submit_pi05_libero_train.sh
```

Or from the LeRobot repo root, with the `lerobot` conda env already active:

```bash
conda activate lerobot
bash examples/LIBERO/train_files/run_pi05_libero_train.sh
```

Useful overrides:

```bash
# Official recipe (8×H100): NUM_PROCESSES=8 BATCH_SIZE=32 STEPS=6000
# This 4×PPU node will auto-fall back to 4 GPUs (effective batch 128).
# Dual-card smoke used BATCH_SIZE=8 @ ~92GB; 32/card may OOM — lower if needed.
STEPS=6000 BATCH_SIZE=32 NUM_PROCESSES=8 \
PI05_PRETRAINED=/mnt/nas/code/models/lerobot_pi05_base \
bash examples/LIBERO/train_files/run_pi05_libero_train.sh
```

The launch script uses `--policy.path` so pretrained camera names are kept. If `config.json` still uses OpenPI names (`base_0_rgb`), it adds `rename_map` automatically. `lerobot/pi05_libero` already matches `observation.images.image` / `image2`.

## Train (π₀.₅ + V-JEPA)

Frozen V-JEPA2 aligns 256 scene queries (after the action suffix) with cosine loss (scale 0.4). `vjepa_align_projector` uses a dedicated `1e-4` LR; the rest of PI0.5 stays at `5e-5`. Sequence layout is `[image | language | action | scene]`. Scene queries attend only to image+language (not action, not other scene tokens); neither the VLM nor the action expert can attend to scene. Scene tokens do not shift action RoPE positions. Float `[0,1]` video clips are passed with `do_rescale=false`. The teacher is loaded lazily for alignment and is neither loaded nor used during action inference.

```bash
bash /mnt/nas/code/lerobot/examples/LIBERO/train_files/submit_pi05_libero_vjepa_train.sh
```

Or from the repo root:

```bash
conda activate lerobot
bash examples/LIBERO/train_files/run_pi05_libero_vjepa_train.sh
```

Overrides:

```bash
STEPS=30000 BATCH_SIZE=32 NUM_PROCESSES=8 \
PI05_PRETRAINED=/mnt/nas/code/models/lerobot_pi05_base \
VJEPA_MODEL_PATH=/mnt/nas/code/starVLA/playground/Pretrained_models/vjepa2-vitl-fpc64-256 \
VJEPA_USE_VIDEO_CLIP=true \
VJEPA_ALIGN_LOSS_SCALE=0.4 \
VJEPA_ALIGN_PROJECTOR_LR=1e-4 \
bash examples/LIBERO/train_files/run_pi05_libero_vjepa_train.sh
```

`VJEPA_USE_VIDEO_CLIP=false` repeats the current primary frame 8 times (bring-up only).

## Resume

```bash
RUN_ID=0813_libero_pi05 bash examples/LIBERO/train_files/run_pi05_libero_resume.sh
RUN_ID=0823_pi05_base_libero_vjepa_v3 bash examples/LIBERO/train_files/run_pi05_libero_vjepa_resume.sh
```

### PI05 单 teacher：VGGT-Ω / LocateAnything

新增的两套配置复现 `starVLA-single` 当前主相机实验，每次只能启用
`vjepa_enabled`、`vggtomega_enabled`、`locateanything_enabled` 中的一个。
VGGT 使用从当前帧开始的 8 帧、224px 输入，对齐 24 层的首帧 patch 特征（196 tokens）；
LocateAnything 使用当前主相机图像和原始 LIBERO 任务抽取的物体名称，对齐 36 层的融合 image tokens。
后者 token 数由 teacher processor 根据图像网格确定，不固定为 V-JEPA 的 256 tokens。

两者默认 uniform 逐层 cosine 对齐，projector LR 为 `1e-4`，target norm 关闭。
VGGT 的 loss scale 为 `0.05`，scene 仅能读取图像；LocateAnything 的 loss scale 为 `0.1`，
scene 可以读取图像和原始任务文本，但不能读取状态数字、action 或其他 scene tokens。
scene 位于 action 后，不改变 action RoPE 位置。Teacher 冻结且不写入 checkpoint；动作推理无需 teacher 文件。

```bash
# 在标准 uv 环境安装；本机已有 PPU PyTorch 环境时请保留厂商 torch 安装。
uv sync --locked --extra pi05-teachers

# 训练入口；YAML 是配套参数说明，实际参数由对应 shell 脚本传给 CLI。
bash examples/LIBERO/train_files/run_pi05_libero_vggtomega_train.sh
bash examples/LIBERO/train_files/run_pi05_libero_locateanything_train.sh

# 单卡短验证示例，输出到新的目录。
CUDA_VISIBLE_DEVICES=0 NUM_PROCESSES=1 STEPS=2 BATCH_SIZE=1 NUM_WORKERS=0 \
  WANDB_ENABLE=false GRAD_CONFLICT_ENABLED=false DATASET_EPISODES='[0]' SAVE_CHECKPOINT=false \
  OUTPUT_DIR=/tmp/pi05_vggtomega_smoke \
  bash examples/LIBERO/train_files/run_pi05_libero_vggtomega_train.sh
```

VGGT 源码路径通过 `VGGTOMEGA_SOURCE_PATH` 配置，默认
`/mnt/nas/code/starVLA/third_party/vggtomega`；权重路径通过 `VGGTOMEGA_CHECKPOINT_PATH` 配置。
仅导入该目录下的 `vggt_omega` 推理包，无需安装 starVLA。可用 `VGGTOMEGA_NUM_FRAMES`
和 `VGGTOMEGA_IMAGE_SIZE` 改变视频长度及输入大小。
LocateAnything 通过 `LOCATEANYTHING_MODEL_PATH` 指定本地 HF 权重目录，默认
`/mnt/nas/code/starVLA/playground/Pretrained_models/LocateAnything-3B`。
使用其自定义模型代码；LeRobot 加载适配层处理 Transformers 5.x 接口差异，不修改共享权重目录。

两套脚本均支持 `PI05_PRETRAINED`、`LIBERO_DATA_ROOT`、`OUTPUT_DIR`、`STEPS`、`BATCH_SIZE`、
`NUM_PROCESSES`，以及 `<VGGTOMEGA|LOCATEANYTHING>_ALIGN_LOSS_SCALE` 和
`<VGGTOMEGA|LOCATEANYTHING>_ALIGN_PROJECTOR_LR`。
短验证可以使用 `DATASET_EPISODES='[0]'` 只读取本地 episode 0，
并设置 `SAVE_CHECKPOINT=false` 避免写入大型优化器状态。
集群入口为对应的 `submit_pi05_libero_vggtomega_train.sh` 和
`submit_pi05_libero_locateanything_train.sh`；仅在需要正式训练时手动提交。

本机已使用 pi05_base 和两个 teacher 的真实权重验证：LIBERO batch size 1 的两步优化、
两套运行脚本的单步训练、冻结 teacher 无梯度、scene/projector 梯度正常。
使用相同 teacher 模型和输入，与 `starVLA-single` 参考特征提取方法逐层比较，
特征及 cosine 对齐损失完全一致；固定噪声时，加入 scene 不改变 action 输出。
真实 policy checkpoint 保存后，已验证 teacher 路径不可用时仍能重载并得到相同动作，
恢复 teacher 路径后可继续计算损失和反向传播。

相关回归测试：

```bash
uv run pytest tests/policies/pi0_pi05/test_pi05.py \
  tests/policies/pi0_pi05/test_pi05_teachers.py \
  tests/policies/pi0_pi05/test_pi05_grad_diagnostics.py \
  tests/processor/test_pi05_processor.py \
  tests/processor/test_pi05_teacher_processor.py -q
```
