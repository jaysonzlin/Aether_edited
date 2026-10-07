# Aether Wan-Style Training Features Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the non-validation training conveniences from Wan’s SimGen path and target/generated depth-like rollout videos to Aether while preserving the current four-GPU, one-sample-per-GPU, 10,000-step run.

**Architecture:** Extend the strict Aether config with Wan-style AdamW settings, constant LR warmup, clipping, data-loader tuning, and optional experiment tracking. Prepare the LR scheduler with Accelerate so checkpoint save/resume includes its state, while disabling Accelerate's per-process scheduler advancement and stepping it explicitly once per synchronized optimizer update. Show progress only on rank zero for optimizer steps, rollout denoising, and the existing model/data fingerprint scan. At each visualization checkpoint, decode both RGB and the 16-channel normalized-disparity output and save paired target/generated MP4s.

**Tech Stack:** Python, PyTorch, Diffusers, Accelerate, Transformers, tqdm, Weights & Biases, pytest.

**Spec:** `docs/superpowers/plans/2026-10-07-fixed-view-simgen-training-bug-fixes.md` plus the user-approved in-chat design in this conversation.

## Global Constraints

- Do not add validation or validation data; continue training all SimGen IDs 0–127.
- Preserve one sample per GPU, four-GPU single-node DDP, bf16, Aether Stage-1 objective, 10,000 optimizer steps, and checkpoint/rollout interval 1,000.
- Add Wan video AdamW hyperparameters: betas `(0.9, 0.95)`, epsilon `1e-8`, weight decay `0.1`; retain learning rate `1e-5`.
- Add constant LR schedule with 200 warmup steps and gradient clipping at max norm `2.0`.
- Set training DataLoader defaults to four workers and pinned memory without changing batch size.
- Enable configurable W&B logging for loss, learning rate, and gradient norm; allow `report_to: null` to disable it.
- Show tqdm progress only on rank zero for training, rollout denoising, and model/data content fingerprinting.
- Save target and generated normalized-disparity videos alongside the existing RGB MP4s every 1,000 steps. Use Aether's inference transform (`mean RGB channels`, map from `[-1,1]` to `[0,1]`, square), then apply Matplotlib's `viridis` colormap to both; label files as disparity/relative-depth visualization, not metric depth.
- `tqdm` is already in `requirements.txt`; add only `wandb>=0.19.0` as a new dependency.
- Include optimizer/scheduler/logging/dataloader settings in the checkpoint run manifest and bump its schema so earlier optimizer-state checkpoints are rejected.

## Review Focus

- Confirm the warmup scheduler advances only after synchronized optimizer updates and resumes from saved scheduler state.
- Confirm clipping runs only at synchronized updates, before `optimizer.step`, and logs the returned pre-clip norm.
- Confirm W&B can be disabled and that only rank zero logs scalar metrics.
- Confirm all progress bars are disabled on non-main ranks and do not wrap validation.
- Confirm DataLoader tuning does not alter the per-GPU batch size or sample set.
- Confirm disparity videos decode latent channels 16–31 and apply the same transformation as `AetherV1PipelineCogVideoX`, then apply the same `viridis` scale to target and generated videos.

## Task 1: Add explicit training configuration and dependency

**Files:** `configs/train/fixed_view_simgen_4h200.yaml`, `aether/training/config.py`, `requirements.txt`, `tests/training/test_training_config.py`.

- [x] Add config fields for `adam_beta1: 0.9`, `adam_beta2: 0.95`, `adam_epsilon: 1.0e-8`, `weight_decay: 0.1`, `warmup_steps: 200`, `max_grad_norm: 2.0`, `dataloader_num_workers: 4`, `pin_memory: true`, `report_to: wandb`, and `wandb_project: aether-fixed-view-simgen`.
- [x] Test loading these defaults, accepting `report_to: null`, and rejecting negative warmup/clipping/worker values or invalid optimizer settings.
- [x] Implement dataclass fields and strict validation while preserving existing fixed-run constants and one-sample local batch.
- [x] Add `wandb>=0.19.0` to `requirements.txt`; do not duplicate the existing tqdm dependency.
- [x] Run `tests/training/test_training_config.py` (12 passed).

## Task 2: Add scheduler, optimizer settings, clipping, and tracker metrics

**Files:** `scripts/train_fixed_view_simgen.py`, `aether/training/checkpointing.py`, `tests/training/test_fixed_view_simgen.py`, `tests/training/test_checkpointing.py`.

- [x] Test construction of the constant-with-warmup schedule, optimizer hyperparameters, synchronized clipping order, and W&B-disabled behavior using focused helpers/fakes.
- [x] Build AdamW from the configured betas, epsilon, and weight decay; build Transformers `get_constant_schedule_with_warmup` with `warmup_steps=200` and the 10,000-step total. Disable Accelerate's default per-process scheduler advancement and step explicitly once per synchronized optimizer update.
- [x] Pass the LR scheduler through `accelerator.prepare` so `save_state`/`load_state` captures its state; advance it only with synchronized optimizer updates.
- [x] Clip gradients to `max_grad_norm` immediately before optimizer updates and capture the returned norm for metrics.
- [x] Initialize Accelerate tracking only when `report_to` is set; log scalar train loss, LR, and gradient norm on the main process.
- [x] Add these settings to run-manifest identity and bump `schema_version` to `2` so checkpoints from prior optimizer semantics are rejected.
- [x] Run focused trainer/checkpoint tests (15 passed).

## Task 3: Add rank-zero progress bars and loader tuning

**Files:** `scripts/train_fixed_view_simgen.py`, `aether/training/rollout.py`, `aether/training/checkpointing.py`, `aether/training/visualization.py`, `tests/training/test_fixed_view_simgen.py`, `tests/training/test_rollout.py`, `tests/training/test_checkpointing.py`, `tests/training/test_visualization.py`.

- [x] Test rank-zero-only train and rollout progress behavior and that fingerprinting can report progress without affecting the resulting manifest.
- [x] Construct the training DataLoader with configured `num_workers` and `pin_memory`, retaining the existing local batch size.
- [x] Add a rank-zero tqdm step bar with current loss, LR, and gradient norm; initialize it at the restored optimizer step and close it cleanly.
- [x] Add an optional tqdm bar over rollout denoising timesteps; keep it disabled unless the caller is main process.
- [x] Add an optional tqdm progress wrapper to the existing rank-zero model/data fingerprint scan.
- [x] Add tests for conversion of `[B,C,F,H,W]` decoded disparity to viridis-colored normalized-disparity frames and for deterministic target/generated disparity MP4 filenames.
- [x] Decode predicted latent channels `16:32`; convert target and prediction using Aether's pipeline convention and Matplotlib `viridis`; save `step_<step>_target_disparity.mp4` and `step_<step>_generated_disparity.mp4` with the RGB artifacts.
- [x] Run focused loader, rollout, and checkpoint tests (15 passed).

## Task 4: Integrated regression and dependency handoff

**Files:** all files above and `docs/superpowers/plans/2026-10-07-aether-wan-training-features.md`.

- [x] Run all `tests/training` tests, CLI help, config dry-run if the local data is complete, Python compilation, and `bash -n submit_fixed_view_simgen_4gpu_mamba.sh` (58 tests passed; dry-run stops at missing local `sample_2/view_0`).
- [x] Inspect the final diff to verify validation remains absent, global batch stays four, and scheduler/checkpoint resume semantics remain consistent.
- [x] Record the new dependency list and any environment/authentication requirements for W&B.
- [x] Report cluster-only checks that still need the H200 environment; do not submit a training job.

## Dependency Handoff

- New: `wandb>=0.19.0` — experiment tracking when `report_to: wandb`.
- Already present: `tqdm>=4.67.1` — progress bars.
- Already present: `accelerate`, `transformers`, and `torch` — distributed tracking, LR scheduler factory, optimizer, and gradient clipping.

W&B setup: authenticate the training environment with `wandb login` or `WANDB_API_KEY`; use `report_to: null` (or `--override report_to=null`) to disable tracking. The local checkout does not contain all 128 SimGen samples, so the full dry run and H200 training check remain to be run on the cluster.
