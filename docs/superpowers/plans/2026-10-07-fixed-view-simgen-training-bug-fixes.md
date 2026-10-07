# Fixed-View SimGen Training Bug-Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct the fixed-view SimGen training objective and make smoke testing, step accounting, checkpoint resume, model preflight, and Slurm log startup reliable.

**Architecture:** Keep the existing Accelerate/DDP training path and its one-sample-per-GPU setup. Derive the diffusion target from the loaded scheduler's configured prediction type, count progress only when the optimizer synchronizes, and save a versioned run manifest with each checkpoint. Validate that manifest before restoring state. Strengthen local model-file preflight and ensure the Slurm output directory exists before Slurm opens its output files.

**Tech Stack:** Python, PyTorch, Diffusers, Accelerate, PyYAML, pytest, Slurm, Bash.

**Spec path:** `docs/superpowers/specs/2026-10-05-fixed-view-simgen-aether-training-design.md` (the prediction-target wording in this older spec is superseded by the actual loaded scheduler configuration described below).

## Global Constraints

- Keep the four-GPU DDP batch at one sample per GPU (global batch four) with the current accumulation setting of one; if supporting larger accumulation, count optimizer updates rather than microbatches.
- Train all 128 SimGen samples with no validation split.
- Fine-tune the entire Aether transformer; preserve the existing model, data, and visualization interfaces except where required by these fixes.
- Use the loaded CogVideoX scheduler's `prediction_type` as the source of truth. For the configured `v_prediction` model, train against `scheduler.get_velocity(target_latents, noise, timesteps)`, not raw noise. Fail clearly on an unsupported prediction type rather than silently selecting a target.
- A checkpoint created under the old raw-noise objective is not compatible with the corrected objective. Reject missing or mismatched run metadata on resume, including legacy checkpoints; do not silently continue from stale optimizer/model state.
- The GPU smoke test must start a fresh one-update run and must not implicitly resume the latest full checkpoint.
- Do not add validation data, change batch sizing, or broaden this into signal-triggered checkpointing.

## Review Focus

- Confirm `v_prediction` targets and any supported `epsilon` targets use the scheduler's exact target semantics and tensor shapes.
- Confirm gradient accumulation increments `global_step`, checkpoint cadence, and max-step termination only on synchronized optimizer updates.
- Confirm smoke-test arguments cannot select `latest` implicitly and execute exactly one fresh optimizer update.
- Confirm resume compatibility covers objective/schema version and training-relevant model, data, and config identity, while excluding ephemeral paths or runtime values that should not prevent relocation.
- Confirm local model checks reject absent or incomplete Aether/CogVideoX directories before loading and Slurm log paths exist before submission.

## Task 1: Correct scheduler-aware diffusion targets

**Files:** `scripts/train_fixed_view_simgen.py`, `tests/training/test_fixed_view_simgen.py` (or the existing focused trainer test module).

- [x] Add focused tests for the configured `v_prediction` target, any intentionally supported `epsilon` target, and unsupported prediction types.
- [x] Implement target selection from the loaded scheduler configuration; use `scheduler.get_velocity(...)` for velocity prediction and raw noise only for epsilon prediction.
- [x] Keep loss over the existing predicted/target latent tensors and preserve the current masking/conditioning behavior.
- [x] Run the focused objective tests and record the result.

## Task 2: Make optimizer-step accounting and smoke-test behavior exact

**Files:** `scripts/train_fixed_view_simgen.py`, `tests/training/test_fixed_view_simgen.py`, `tests/training/test_training_config.py` as needed.

- [x] Add a regression test showing a smoke test performs one fresh optimizer update even when a completed checkpoint exists in the output directory.
- [x] Add a regression test for gradient accumulation showing global step and max-step/output cadence advance only when gradients synchronize and the optimizer updates.
- [x] Make `--gpu-smoke-test` reject or override resume explicitly and ensure it executes one optimizer update from freshly initialized optimizer state.
- [x] Move global-step progression and step-boundary actions to synchronized optimizer-update boundaries without changing one sample per GPU/global batch four.
- [x] Run the focused trainer/config tests.

## Task 3: Version checkpoint state and reject incompatible resumes

**Files:** `aether/training/checkpointing.py`, `scripts/train_fixed_view_simgen.py`, `tests/training/test_checkpointing.py`, trainer tests.

- [x] Add tests for writing and reading a run manifest, accepting a compatible checkpoint, and rejecting missing, legacy, or mismatched metadata with an actionable error.
- [x] Define a manifest schema/version that captures the objective/prediction type and the training-relevant model, dataset, and config identity.
- [x] Save the manifest alongside each checkpoint's Accelerator state and global step.
- [x] Validate compatibility before loading model/optimizer state for explicit and `latest` resume paths; make `latest` skip or fail clearly on incompatible checkpoints rather than loading them.
- [x] Run checkpointing and resume tests, including a legacy checkpoint fixture.

## Task 4: Strengthen model preflight and Slurm log-directory availability

**Files:** `aether/training/config.py` or the existing preflight module, `submit_fixed_view_simgen_4gpu_mamba.sh`, `.gitignore`, `logs/.gitkeep`, `tests/training/test_training_config.py`, `tests/training/test_slurm_launch.py`.

- [x] Add tests showing empty/incomplete local model directories fail preflight with the missing expected artifact identified, while valid local layouts pass.
- [x] Check for the essential local Aether and CogVideoX config/weight artifacts before model construction; do not download weights implicitly.
- [x] Ensure the `logs/` directory is present in a fresh checkout before Slurm resolves `#SBATCH --output`/`--error` paths. Keep generated logs ignored while tracking only a directory sentinel.
- [x] Add/adjust launch-script tests to verify output/error directories are present at submission time and retain the configured cluster paths.
- [x] Run focused preflight and launch tests plus `bash -n submit_fixed_view_simgen_4gpu_mamba.sh`.

## Task 5: Integrated regression pass and handoff

**Files:** affected tests and implementation files from Tasks 1–4.

- [x] Run the focused training test suite and then all `tests/training` tests; run CLI help and dry-run checks if they do not require unavailable model weights or GPUs.
- [x] Review the final diff for accidental batch-size, validation, checkpoint-retention, or rollout behavior changes.
- [x] Summarize the implemented fixes, tests actually run, and any cluster-only checks still requiring the H200 environment.

## Deferred Risk

Slurm requeue still does not trigger an immediate checkpoint on a pre-timeout signal, so a requeued job may lose work since its most recent periodic save. This is intentionally deferred from this reliability pass.

## Execution Notes

- `python -m pytest tests/training -q`: 41 passed.
- Rank zero computes the content hashes once and broadcasts the run manifest to the other DDP ranks.
- CLI help, Python compilation, Slurm `bash -n`, and `git diff --check` passed.
- The default-data dry run was attempted but cannot validate the local sibling dataset because `../simgen/runs/panda_ball_can/sample_2/view_0` is missing. The cluster dataset still needs its own dry run/preflight.
- No H200 GPU smoke test was run from this workstation.
