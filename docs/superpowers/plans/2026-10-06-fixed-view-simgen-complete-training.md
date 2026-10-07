# Fixed-View SimGen Complete Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver a stable, resumable four-H200 DDP Aether fixed-view SimGen training path, including real fixed-seed rollout MP4s every 1,000 steps.

**Architecture:** Preserve the Aether 41-frame RGB-D/raymap contract and full-transformer Stage-1 fine-tuning. The trainer uses Accelerator DDP, a frozen VAE/text encoder, numbered synchronized checkpoints, and a separate Aether-compatible inference scheduler for rank-zero fixed-sample visualizations.

**Tech Stack:** Python, PyTorch, Diffusers, Accelerate, h5py, imageio, PyYAML, pytest, Slurm, Mamba.

## Global Constraints

- Use all 128 `sample_<id>/view_0` SimGen records; no validation split.
- Use 41 source frames at 480×720 after 120-pixel horizontal padding, four latent-history slots, and 13 observed raw RGB frames in saved rollouts.
- Fine-tune only the Aether transformer; VAE and text encoder stay frozen.
- Train Stage 1 noise MSE only for 10,000 optimizer steps in bf16 four-GPU DDP.
- Use local model directories under `/n/lab_storage/ydu_lab/jaysonzlin/Aether_edited/models` in the Slurm launch path.
- Save synchronized checkpoints and fixed `sample_0`/seed-42 target/generated MP4s at steps 1,000 through 10,000; retain two checkpoints.
- Commit and push every verified task.

## Task 1: Reset and harden rollout primitives

**Files:** `aether/training/rollout.py`, `tests/training/test_rollout.py`

- [x] Remove the uncommitted sampler draft and rebuild test-first.
- [x] Test and implement four-history unconditional conditioning, exact Aether dynamic CFG scale 3.0, strict 41-frame checks, and 13-frame prefix compositing.
- [x] Test scheduler isolation and seeded deterministic latent sampling with fake transformer/scheduler components.
- [x] Run `pytest tests/training/test_rollout.py -v`; commit and push.

## Task 2: Implement real fixed rollout sampling

**Files:** `aether/training/rollout.py`, `tests/training/test_rollout.py`

- [x] Implement Aether’s 50-step CogVideoX DPM denoising loop using a cloned inference scheduler.
- [x] Use CFG by concatenating unconditional and conditional states; zero only the four RGB-history slots in the unconditional state and preserve raymaps.
- [x] Require target shape `[B, 11, 56, 60, 90]` and condition shape `[B, 11, 40, 60, 90]`.
- [x] Run focused tests; commit and push.

## Task 3: Integrate checkpoint, resume, and rollout artifacts

**Files:** `scripts/train_fixed_view_simgen.py`, `aether/training/checkpointing.py`, `aether/training/visualization.py`, tests under `tests/training/`

- [x] Save/restore Accelerator state and global step, skip malformed checkpoints, and retain two checkpoint directories.
- [x] On every 1,000-step boundary, synchronize ranks, save checkpoint, and run visualization only on rank zero.
- [x] Load fixed `sample_0`, encode it with the frozen VAE, sample with seed 42, decode RGB, require 41 frames, composite the 13 observed frames, and save target/generated MP4s.
- [x] Restore transformer training mode and synchronize all ranks before the next step.
- [x] Run focused tests and `pytest tests/training -v`; commit and push.

## Task 4: Complete trainer safety and GPU preflight

**Files:** `scripts/train_fixed_view_simgen.py`, `aether/training/config.py`, tests under `tests/training/`

- [x] Validate local model directory layout, data manifest, CUDA/bf16 availability, and four visible GPUs without beginning training.
- [x] Keep `--gpu-smoke-test` as one real finite optimizer step and make completed runs exit without step 10,001.
- [ ] Run CLI help, dry-run, fake/CPU tests, and cluster GPU smoke test when available; commit and push.

## Task 5: Finalize four-H200 Slurm/Mamba launch

**Files:** `configs/accelerate/h200_4gpu.yaml`, `submit_fixed_view_simgen_4gpu_mamba.sh`, `tests/training/test_slurm_launch.py`, documentation

- [x] Use one node, four H200s, 24-hour requeue, Wan-style modules/NCCL diagnostics, and `/n/holylabs/ydu_lab/Lab/jaysonzlin/aether_env`.
- [x] Pass explicit local Aether/CogVideoX paths and `--resume latest`; use no multi-node `srun` fan-out.
- [x] Document setup, preflight, smoke, `sbatch`, checkpoint locations, and no-validation rationale.
- [ ] Run `bash -n`, all training tests, and launch-script static tests; commit and push.
