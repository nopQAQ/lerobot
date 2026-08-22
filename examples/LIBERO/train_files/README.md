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

V-JEPA prefix scene-token recipe (same OpenPI hyperparameters, alignment on):

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

Frozen V-JEPA2 aligns 256 prefix scene queries with cosine loss (scale 0.5). Scene queries attend only to image+language; neither the VLM nor the action expert can attend to scene. Scene tokens do not shift action RoPE positions. Float `[0,1]` video clips are passed with `do_rescale=false`. The teacher is loaded lazily for alignment and is neither loaded nor used during action inference.

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
bash examples/LIBERO/train_files/run_pi05_libero_vjepa_train.sh
```

`VJEPA_USE_VIDEO_CLIP=false` repeats the current primary frame 8 times (bring-up only).

## Resume

```bash
RUN_ID=0813_libero_pi05 bash examples/LIBERO/train_files/run_pi05_libero_resume.sh
RUN_ID=0819_pi05_base_libero_vjepa_v2 bash examples/LIBERO/train_files/run_pi05_libero_vjepa_resume.sh
```
