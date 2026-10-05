# Fixed-View SimGen Aether Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a testable Stage-1 training path that full-fine-tunes Aether on padded, fixed-view SimGen RGB-D clips.

**Architecture:** Add a focused `aether.training` package for sample validation/preprocessing, Aether latent assembly, and checkpointed training. Keep the released inference pipeline intact; the new trainer calls the same frozen CogVideoX components but owns its training-only flow.

**Tech Stack:** Python 3.10, PyTorch, Diffusers, Accelerate, h5py, NumPy, PyYAML, pytest.

**Spec:** `docs/superpowers/specs/2026-10-05-fixed-view-simgen-aether-training-design.md`

## Global Constraints

- Consume only SimGen-style `sample_<id>/view_0` records; validate all 41 required frames.
- Use 480x720 padded clips, 41 frames, four history latent positions, and empty text context.
- Freeze VAE and text encoder; make only the Aether transformer trainable.
- Stage 1 is latent-space noise MSE only; do not add point-cloud, validation, or decoded-image losses.
- Visualize and checkpoint every 1,000 steps through step 10,000.

## Review Focus

- A malformed depth or camera file must fail before CUDA/model initialization.
- Padding must shift `cx` exactly +120 without modifying `fx` or `fy`.
- Depth normalization must ignore invalid pixels and reject clips without valid depth.
- Raymap packing must preserve 41 source frames in the 11-latent-time representation.
- Resume must not restart global-step numbering or overwrite an existing checkpoint.

---

## File Structure

- Create `aether/training/simgen_dataset.py`: validates and loads padded RGB-D-camera clips.
- Create `aether/training/geometry.py`: disparity conversion, intrinsic padding, raymap construction and temporal packing.
- Create `aether/training/aether_latents.py`: frozen VAE encoding and 56/40-channel Aether state assembly.
- Create `aether/training/config.py`: strict YAML config loading and validation.
- Create `scripts/train_fixed_view_simgen.py`: distributed Stage-1 trainer, visualization, resume, and checkpoint retention.
- Create `configs/train/fixed_view_simgen_4h200.yaml`: portable first-run configuration.
- Create focused tests under `tests/training/`.

### Task 1: SimGen Dataset and Geometry Contract

**Files:**
- Create: `aether/training/simgen_dataset.py`
- Create: `aether/training/geometry.py`
- Create: `tests/training/test_simgen_dataset.py`
- Create: `tests/training/test_geometry.py`

**Interfaces:**
- Produces `SimGenFixedViewDataset(root: str | Path, sample_ids: Sequence[int])` whose item has `rgb`, `disparity`, `raymap`, `content_mask`, and `sample_id`.
- Produces `pad_intrinsics(camera: Mapping[str, object], left_padding: int = 120) -> np.ndarray` and `pack_raymaps(raymaps: np.ndarray) -> np.ndarray`.

- [ ] Write failing tests for valid 41-frame loading, a missing frame, padded dimensions, `cx + 120`, invalid depth rejection, and packed raymap shape `(11, 24, 60, 90)`.
- [ ] Run `pytest tests/training/test_simgen_dataset.py tests/training/test_geometry.py -v`; confirm the failures are from missing training modules.
- [ ] Implement the minimal validated dataset and geometry helpers, including the specified edge/far-depth padding policy and paper-compatible disparity transformation.
- [ ] Re-run the focused tests until they pass.
- [ ] Commit the dataset and geometry contract.

### Task 2: Aether Latent Assembly

**Files:**
- Create: `aether/training/aether_latents.py`
- Create: `tests/training/test_aether_latents.py`

**Interfaces:**
- Consumes a batch returned by `SimGenFixedViewDataset` and injected frozen VAE encoder.
- Produces `AetherTrainingBatch(target_latents, condition_latents, history_positions)` with target shape `[B, 11, 56, 60, 90]` and condition shape `[B, 11, 40, 60, 90]`.

- [ ] Write failing tests with a deterministic fake VAE for modality order, four history condition positions, zero-filled later RGB conditions, and always-present raymap conditions.
- [ ] Run `pytest tests/training/test_aether_latents.py -v`; confirm the expected import/API failure.
- [ ] Implement latent assembly without constructing a real model in unit tests.
- [ ] Re-run focused tests and commit the latent assembly.

### Task 3: Configuration and Trainer Smoke Path

**Files:**
- Create: `aether/training/config.py`
- Create: `configs/train/fixed_view_simgen_4h200.yaml`
- Create: `scripts/train_fixed_view_simgen.py`
- Create: `tests/training/test_training_config.py`

**Interfaces:**
- Consumes one YAML config and optional key-value overrides.
- Produces a validated config with exact first-run constants: 41 frames, 480x720, 4 history slots, 10,000 steps, 1,000-step output cadence, bf16, and all 128 sample IDs.

- [ ] Write failing config tests for exact first-run values and invalid temporal/padding combinations.
- [ ] Run `pytest tests/training/test_training_config.py -v`; confirm it fails because the config loader is absent.
- [ ] Implement strict config validation and a CLI that loads Aether/CogVideoX components, freezes support modules, constructs the data loader, and executes one optimizer step under Accelerate.
- [ ] Add a CPU/fake-component smoke mode proving finite Stage-1 MSE without downloading weights.
- [ ] Run focused tests and commit the trainer smoke path.

### Task 4: Checkpoint, Resume, and Fixed Visualization

**Files:**
- Modify: `scripts/train_fixed_view_simgen.py`
- Create: `aether/training/checkpointing.py`
- Create: `aether/training/visualization.py`
- Create: `tests/training/test_checkpointing.py`
- Create: `tests/training/test_visualization.py`

**Interfaces:**
- Produces `save_checkpoint(...)`, `restore_checkpoint(...)`, and `save_fixed_rollout(...)`.
- Consumes trainer state and a fixed sample/seed; emits numbered checkpoints and target/generated MP4 artifacts.

- [ ] Write failing tests for preserving global step on restore, checkpoint retention, and output filenames at 1,000-step cadence.
- [ ] Run focused tests and confirm expected missing-module failures.
- [ ] Implement minimal checkpoint/visualization helpers and integrate them into the trainer.
- [ ] Run focused tests, then `pytest tests/training -v`, and commit.

### Task 5: Four-H200 Preflight and Documentation

**Files:**
- Modify: `README.md`
- Create: `docs/fixed_view_simgen_training.md`
- Modify: `configs/train/fixed_view_simgen_4h200.yaml`
- Test: `tests/training/`

- [ ] Document required model downloads, the SimGen directory contract, four-GPU launch command, checkpoint resume command, and why no validation is used initially.
- [ ] Add a dry-run command that validates all 128 samples before model loading.
- [ ] Run `pytest tests/training -v` and `python scripts/train_fixed_view_simgen.py --help`.
- [ ] Commit the launch documentation and preflight support.
