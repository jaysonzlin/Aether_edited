# Fixed-View SimGen Training

This is the first Aether fixed-view dynamics experiment: full Stage-1
fine-tuning of the transformer on all 128 SimGen records. It deliberately has
no validation split while training stability is established.

## Data and model layout

Each record must be `sample_<id>/view_0` beneath the configured SimGen root,
with 41 RGB/depth frames and its camera metadata. The dataset pads each
480x480 RGB-D sequence horizontally to 480x720, packs 11 raymap latent times,
and uses four RGB latent-history slots (13 raw observed RGB frames in previews).

The default configuration assumes sibling `Aether`, `Wan2.2_edited`, and
`simgen` directories. It uses `../simgen/runs/panda_ball_can` from the Aether
repository. Override `data_root` when the directory lives elsewhere.

Download full snapshots, not the Hugging Face cache directory, into the model
layout expected by the trainer:

```console
hf download AetherWorldModel/AetherV1 \
  --local-dir /n/lab_storage/ydu_lab/jaysonzlin/Aether_edited/models/AetherV1
hf download zai-org/CogVideoX-5b-I2V \
  --local-dir /n/lab_storage/ydu_lab/jaysonzlin/Aether_edited/models/CogVideoX-5b-I2V
```

The required subdirectories are:

```text
models/AetherV1/transformer/
models/CogVideoX-5b-I2V/{tokenizer,text_encoder,vae,scheduler}/
```

Training never downloads checkpoints automatically. The Slurm launcher uses
offline Hugging Face mode, so both snapshots must exist before submission.

## Preflight and smoke test

Run preflight from a four-H200 allocation. It validates the model layout,
every one of the 128 data records, CUDA bf16 support, and exactly four visible
GPUs without loading model weights or beginning optimization.

```console
/n/holylabs/ydu_lab/Lab/jaysonzlin/aether_env/bin/python \
  scripts/train_fixed_view_simgen.py --preflight \
  --override aether_model_id=/n/lab_storage/ydu_lab/jaysonzlin/Aether_edited/models/AetherV1 \
  --override cogvideox_model_id=/n/lab_storage/ydu_lab/jaysonzlin/Aether_edited/models/CogVideoX-5b-I2V
```

To test the actual CUDA model path and one finite optimizer update, use one
GPU. This is separate from four-GPU preflight:

```console
CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 \
  scripts/train_fixed_view_simgen.py --gpu-smoke-test
```

`--dry-run` is CPU-only and validates the dataset contract; it does not need
checkpoints or a GPU.

## Launch and resume

Submit the standard one-node, four-H200 DDP job from the repository root:

```console
sbatch submit_fixed_view_simgen_4gpu_mamba.sh
```

The launcher activates
`/n/holylabs/ydu_lab/Lab/jaysonzlin/aether_env`, uses `MULTI_GPU` Accelerate
DDP (no `srun` fan-out), explicitly supplies the local Aether and CogVideoX
paths, requests a requeueable 24-hour H200 allocation, and resumes the newest
checkpoint automatically.

The job runs 10,000 optimizer steps in bf16. It writes two resumable
Accelerate checkpoints at a time under `outputs/fixed_view_simgen/`, retaining
the newest two numbered directories. Every 1,000 steps it additionally writes
both `step_<step>_target.mp4` and `step_<step>_generated.mp4`. The generated
MP4 is a real 50-step Aether DPM rollout of `sample_0` at seed 42; its first
13 raw RGB frames are replaced with the observed history for visual alignment.

## Training metrics

With W&B enabled, the trainer logs `train/loss`, `train/learning_rate`,
`train/grad_norm`, and per-output-slice losses: `train/rgb_loss`,
`train/disparity_loss`, and `train/raymap_loss`. These are per-element MSEs
against the same scheduler-derived diffusion target, split across the 16 RGB,
16 normalized-disparity, and 24 raymap latent output channels. They are
diagnostics, not separately weighted objectives; `train/loss` remains the
single optimization objective and equals the channel-count-weighted mean of
the three component losses. The disparity metric is latent-space diffusion
loss, not pixel-space or metric-depth error. No PC predictor loss is included.
