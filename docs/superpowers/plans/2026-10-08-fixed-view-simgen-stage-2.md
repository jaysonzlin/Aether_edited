# Fixed-View SimGen Stage-2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a separate resumable fixed-view SimGen Stage-2 image-space refinement job initialized from completed Stage 1.

**Architecture:** Leave Stage 1 unchanged. A strict Stage-2 config and model-only checkpoint loader own transfer; a focused loss module owns clean-latent reconstruction, decoded losses, and calibration; a new trainer combines them and a new Slurm launcher exposes the job.

**Tech Stack:** Python, PyTorch, Diffusers, Accelerate, torchmetrics, safetensors, PyYAML, pytest, Slurm.

**Spec:** `docs/superpowers/specs/2026-10-08-fixed-view-simgen-stage-2-design.md`

## Global Constraints

- Preserve Stage-1 code, `FixedViewTrainingConfig`, checkpoints, and launcher behavior.
- Require `outputs/fixed_view_simgen/checkpoint-010000` and import transformer weights only.
- Use the 128 existing fixed-view clips, 41-frame contract, four H200s, bf16 DDP, batch size one, four history slots, and frozen VAE/text encoder.
- Run exactly 2,500 updates with a fresh AdamW and OneCycleLR at peak `1e-5` and `pct_start=0.1`.
- Optimize full-video MSE plus valid-content-only RGB MS-SSIM, depth SSI, and depth-detached raymap pointmap losses.
- Save Stage-2 checkpoints and fixed previews every 500 updates, retaining two checkpoints.
- Launch offline and preflight local assets, `tiktoken`, decoded-loss imports, CUDA bf16, and four visible GPUs.

## Review Focus

- A missing or partial Stage-1 checkpoint must fail rather than falling back to an earlier state.
- Requeue must reuse calibrated weights and reject a mismatched Stage-2 manifest.
- Frozen VAE parameters must not prevent gradients from flowing through predicted VAE decoder inputs.
- Decoded losses and their gradients must ignore all content outside columns `[120, 600)`.
- Epsilon and velocity schedulers must reconstruct clean latents correctly; unsupported modes fail explicitly.

---

### Task 1: Stage-2 configuration and safe checkpoint transfer

**Files:**
- Create: `aether/training/stage2_config.py`
- Modify: `aether/training/checkpointing.py`
- Create: `tests/training/test_stage2_config.py`
- Modify: `tests/training/test_checkpointing.py`

**Interfaces:**
- Produces `Stage2TrainingConfig`, `load_stage2_training_config(path, overrides=())`, `load_stage1_transformer_weights(transformer, checkpoint)`, and a Stage-2 manifest builder.

- [ ] **Step 1: Write failing tests**

Test the exact 2,500/500/`1e-5`/10% config contract, missing or incomplete `checkpoint-010000`, a source global step other than 10,000, and model-only safetensors loading.

- [ ] **Step 2: Verify RED**

Run `pytest tests/training/test_stage2_config.py tests/training/test_checkpointing.py -v`; expect failure because the Stage-2 interfaces do not exist.

- [ ] **Step 3: Implement the config and transfer boundary**

Create the strict independent config. Extend checkpointing to validate Stage-1 metadata and load only transformer safetensors into the unwrapped model, reporting key mismatches; never call `Accelerator.load_state` for this transfer. Build a Stage-2 manifest with objective `stage2_image_space_refinement` and its calibrated loss contract.

- [ ] **Step 4: Verify GREEN**

Run `pytest tests/training/test_stage2_config.py tests/training/test_checkpointing.py -v`; expect pass.

- [ ] **Step 5: Commit**

Commit config, checkpointing, and their tests with message `feat: add stage 2 training configuration`.

### Task 2: Decoded loss and calibration primitives

**Files:**
- Create: `aether/training/stage2_losses.py`
- Create: `tests/training/test_stage2_losses.py`

**Interfaces:**
- Produces `reconstruct_clean_latents(...)`, `compute_stage2_losses(...)`, and `calibrate_auxiliary_weights(...)` for Task 3.

- [ ] **Step 1: Write failing tests**

Cover epsilon and velocity reconstruction, unsupported prediction failure, loss invariance to padding-only changes, finite RGB/SSI/pointmap losses, absent depth gradient and nonzero raymap gradient from pointmap loss, 24-to-6 raymap unpacking to 41 frames, and deterministic calibration serialization/reuse.

- [ ] **Step 2: Verify RED**

Run `pytest tests/training/test_stage2_losses.py -v`; expect failure because the loss module is absent.

- [ ] **Step 3: Implement loss primitives**

Implement clean-latent reconstruction; VAE-output RGB/disparity mapping; `torchmetrics` MS-SSIM; stabilized, equal-data-and-gradient L1 SSI; raymap unpacking and upsampling; scale-and-translation-aligned, inverse-target-depth-weighted pointmap L1; and detached predicted disparity in the pointmap path. Calibrate nonzero auxiliary terms to the detached MSE magnitude.

- [ ] **Step 4: Verify GREEN**

Run `pytest tests/training/test_stage2_losses.py -v`; expect pass.

- [ ] **Step 5: Commit**

Commit the module and tests with message `feat: add Aether stage 2 image losses`.

### Task 3: Stage-2 trainer and resumability

**Files:**
- Create: `scripts/train_fixed_view_simgen_stage2.py`
- Modify: `aether/training/checkpointing.py`
- Create: `tests/training/test_fixed_view_simgen_stage2.py`
- Create: `tests/training/test_gpu_stage2_smoke_cli.py`

**Interfaces:**
- Consumes Tasks 1 and 2 plus existing dataset, latent assembly, rollout, and optimizer helpers.
- Produces the `--preflight`, `--gpu-smoke-test`, `--resume latest`, and `--override` CLI and independent Stage-2 state.

- [ ] **Step 1: Write failing trainer tests**

Test initialization before `accelerator.prepare`, fresh AdamW/OneCycle construction, rank-zero calibration broadcast and JSON persistence, reuse on resume, raw/weighted logging keys, final-step checkpointing, and smoke/resume argument rejection.

- [ ] **Step 2: Verify RED**

Run `pytest tests/training/test_fixed_view_simgen_stage2.py tests/training/test_gpu_stage2_smoke_cli.py -v`; expect failure because the CLI is absent.

- [ ] **Step 3: Implement the trainer**

Mirror Stage-1 distributed loading and rollout. Encode targets without grad; decode predicted clean RGB/disparity with decoder autograd; calculate MSE plus calibrated losses; update via fresh OneCycle schedule; and log raw/weighted metrics. Save and validate Stage-2 manifest, optimizer, scheduler, calibration, and progress. Save state and the existing deterministic rollout artifacts at each 500-step boundary, including 2,500.

- [ ] **Step 4: Verify GREEN**

Run `pytest tests/training/test_fixed_view_simgen_stage2.py tests/training/test_gpu_stage2_smoke_cli.py -v`; expect pass.

- [ ] **Step 5: Commit**

Commit trainer, checkpoint updates, and tests with message `feat: add fixed-view stage 2 trainer`.

### Task 4: H200 launch surface and end-to-end verification

**Files:**
- Create: `configs/train/fixed_view_simgen_stage2_4h200.yaml`
- Create: `submit_fixed_view_simgen_stage2_4gpu_mamba.sh`
- Modify: `requirements.txt`
- Modify: `docs/fixed_view_simgen_training.md`
- Create: `tests/training/test_stage2_slurm_launch.py`

**Interfaces:**
- Consumes Task 3's CLI and source checkpoint contract.
- Produces a submit-ready offline Slurm path and operator documentation.

- [ ] **Step 1: Write failing launch tests**

Assert exact YAML values, separate output path, source-checkpoint reference, four-H200 Accelerate command, offline flags, `tiktoken`/loss imports, and no mutation of the Stage-1 launcher.

- [ ] **Step 2: Verify RED**

Run `pytest tests/training/test_stage2_slurm_launch.py -v`; expect failure because Stage-2 assets are absent.

- [ ] **Step 3: Implement operator assets**

Add `tiktoken` to requirements. Add the approved YAML, a mirrored offline launcher that passes explicit local assets and the exact Stage-1 source, and docs for preflight, smoke, submission, requeue, calibration metadata, and artifacts.

- [ ] **Step 4: Verify GREEN and regressions**

Run `pytest tests/training -v && bash -n submit_fixed_view_simgen_stage2_4gpu_mamba.sh && python scripts/train_fixed_view_simgen_stage2.py --help && python -m compileall aether/training scripts/train_fixed_view_simgen_stage2.py`; expect all commands to succeed.

- [ ] **Step 5: Commit**

Commit operator assets and docs with message `feat: add stage 2 H200 launch path`.
