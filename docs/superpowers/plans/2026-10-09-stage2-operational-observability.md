# Stage-2 Operational Observability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the Stage-2 H200 job the same operational safeguards and W&B lifecycle as Stage 1, with scalar telemetry specific to Stage 2's calibrated image-space objective.

**Architecture:** Preserve Stage 2's existing trainer, Python-module Accelerate invocation, preflight boundary, and distinct outputs. The launcher owns Slurm/NCCL diagnostics and executable/dependency checks; the existing `--preflight` remains the sole authority for models, dataset, source checkpoint, and GPU contract. The trainer owns W&B lifecycle, progress reporting, and Stage-2-specific metric payloads.

**Tech Stack:** Bash, Slurm, NCCL, Python, PyTorch, Hugging Face Accelerate, Weights & Biases, pytest.

**Spec:** `docs/superpowers/specs/2026-10-08-fixed-view-simgen-stage-2-design.md`

## Global Constraints

- Do not modify the Stage-1 launcher, Stage-1 trainer, Stage-1 outputs, or model architecture.
- Preserve Stage 2's Python-module launch form: `"${PYTHON_BIN}" -m accelerate.commands.accelerate_cli launch`.
- Keep Stage 2's email recipient and lifecycle notifications: `jlin3@college.harvard.edu`, `BEGIN,END,FAIL`.
- Mirror only Stage-1 operational behavior that is compatible with Stage 2; use Stage-2-specific output and NCCL-log names.
- The shell validates its execution environment; `scripts/train_fixed_view_simgen_stage2.py --preflight` remains the one source of truth for local model files, dataset, completed Stage-1 checkpoint, CUDA bf16 support, and four GPUs.
- W&B remains enabled by default through `report_to: wandb`; `--override report_to=null` disables it without requiring a new launcher mode.
- Log scalars only. Do not upload images, videos, checkpoints, or other W&B artifacts.
- Do not submit a cluster job or contact W&B as part of this implementation.

## Review Focus

- `report_to: null` must skip both W&B initialization and finalization; Task 1 adds a fake-Accelerator test for this path.
- An unsupported tracker or blank project name must fail during configuration load; Task 1 extends config validation tests.
- A run that raises after tracker initialization must still call `end_training()` once without masking the original exception; Task 2 verifies the lifecycle helper/wiring with a recording fake.
- Requeues must create Stage-2-specific NCCL files and append Slurm output rather than overwriting prior attempts; Task 3 asserts both directives and names.
- Calibration values loaded from a previous run must be logged as the effective Stage-2 weights, not the YAML multipliers; Task 2 tests the metric payload with distinct configured and calibrated values.

---

### Task 1: Define testable Stage-2 tracking contracts

**Files:**
- Modify: `aether/training/stage2_config.py:64-110`
- Modify: `scripts/train_fixed_view_simgen_stage2.py:1-130`
- Modify: `tests/training/test_stage2_config.py`
- Modify: `tests/training/test_stage2_smoke_cli.py`

**Interfaces:**
- Produces `initialize_stage2_tracking(accelerator, config) -> bool`, which initializes Accelerate trackers only when `config.report_to` is set and returns whether finalization is required.
- Produces `finish_stage2_tracking(accelerator, initialized: bool) -> None`, which calls `accelerator.end_training()` only for an initialized tracker.
- Produces `stage2_tracker_config(config) -> dict[str, object]`, with the Stage-2 optimizer, schedule, interval, source-checkpoint, and configured auxiliary-loss settings.
- Produces `stage2_metric_values(*, total, mse, losses, learning_rate, grad_norm) -> dict[str, float]` and `stage2_calibration_metric_values(weights) -> dict[str, float]`; the first describes each optimizer update and the second describes the effective calibrated weights once per run/resume.

- [ ] **Step 1: Write failing configuration and tracker tests**

Add these tests before production changes:

```python
@pytest.mark.parametrize("report_to", ["tensorboard", "mlflow"])
def test_stage2_config_rejects_unsupported_tracker(tmp_path, report_to):
    path = tmp_path / "stage2.yaml"
    _write_config(path, report_to=report_to)
    with pytest.raises(Stage2TrainingConfigError, match="report_to"):
        load_stage2_training_config(path)

def test_initialize_stage2_tracking_uses_project_and_stage2_metadata():
    accelerator = RecordingAccelerator()
    assert initialize_stage2_tracking(accelerator, config) is True
    assert accelerator.project == "aether-fixed-view-simgen-stage2"
    assert accelerator.metadata["onecycle_pct_start"] == 0.1
    assert accelerator.metadata["rgb_loss_weight"] == 1.0

def test_initialize_stage2_tracking_skips_disabled_reporting():
    config = replace(config, report_to=None)
    assert initialize_stage2_tracking(RecordingAccelerator(), config) is False
```

Add a pure per-update payload test asserting `train/loss`, `train/mse`, `train/rgb_ms_ssim`, `train/depth_ssi`, `train/pointmap`, `train/learning_rate`, and `train/grad_norm`. Add a separate calibration-payload test with deliberately different calibrated values that asserts `train/calibrated_{rgb,depth,pointmap}_weight`.

- [ ] **Step 2: Verify RED**

Run: `PYTHONPATH=. pytest tests/training/test_stage2_config.py tests/training/test_stage2_smoke_cli.py -v`

Expected: FAIL because tracker validation/helpers and the expanded metrics contract do not exist.

- [ ] **Step 3: Implement the narrow tracking boundary**

In `stage2_config._validate`, accept only `None` and `"wandb"` for `report_to`, and require a nonblank string for `wandb_project`, matching Stage 1.

In `scripts/train_fixed_view_simgen_stage2.py`, implement the three interfaces above. `stage2_tracker_config` must include:

```python
{
    "learning_rate": config.learning_rate,
    "max_train_steps": config.max_train_steps,
    "onecycle_pct_start": config.onecycle_pct_start,
    "adam_beta1": config.adam_beta1,
    "adam_beta2": config.adam_beta2,
    "adam_epsilon": config.adam_epsilon,
    "weight_decay": config.weight_decay,
    "max_grad_norm": config.max_grad_norm,
    "output_interval": config.output_interval,
    "stage1_checkpoint": config.stage1_checkpoint,
    "rgb_loss_weight": config.rgb_loss_weight,
    "depth_loss_weight": config.depth_loss_weight,
    "pointmap_loss_weight": config.pointmap_loss_weight,
}
```

Use `accelerator.init_trackers(config.wandb_project, config=stage2_tracker_config(config))`. Keep both metric helpers pure; `stage2_calibration_metric_values` receives effective `weights`, never recalculates them.

- [ ] **Step 4: Verify GREEN**

Run: `PYTHONPATH=. pytest tests/training/test_stage2_config.py tests/training/test_stage2_smoke_cli.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add aether/training/stage2_config.py scripts/train_fixed_view_simgen_stage2.py \
  tests/training/test_stage2_config.py tests/training/test_stage2_smoke_cli.py
git commit -m "feat: initialize stage 2 experiment tracking"
```

### Task 2: Wire Stage-2 progress, scalar logging, and cleanup

**Files:**
- Modify: `scripts/train_fixed_view_simgen_stage2.py:126-199`
- Modify: `tests/training/test_stage2_smoke_cli.py`

**Interfaces:**
- Consumes `initialize_stage2_tracking`, `finish_stage2_tracking`, `stage2_metric_values`, `stage2_calibration_metric_values`, and the existing `optimizer_update(...) -> Tensor | None` return value.
- Produces one main-process W&B scalar payload per synchronized optimizer update and one calibration-weight payload after weights have been loaded or calibrated.

- [ ] **Step 1: Write failing lifecycle and logging tests**

Add a recording-Accelerator test that verifies:

```python
assert stage2_calibration_metric_values(
    {"rgb": 2.0, "depth": 3.0, "pointmap": 4.0}
)["train/calibrated_pointmap_weight"] == 4.0
```

Add a cleanup test that exercises the small extracted tracking cleanup helper with a recording accelerator and asserts `end_training()` occurs only when initialization succeeded. The test must also verify that a simulated training exception remains the raised exception after cleanup.

- [ ] **Step 2: Verify RED**

Run: `PYTHONPATH=. pytest tests/training/test_stage2_smoke_cli.py -v`

Expected: FAIL because cleanup and full Stage-2 metric wiring are absent.

- [ ] **Step 3: Implement Stage-2-only runtime observability**

After constructing `Accelerator`, call `initialize_stage2_tracking`. Wrap all subsequent training setup and looping in `try/finally`; the `finally` calls `accelerator.end_training()` only when tracking was initialized.

Mirror Stage 1's rank-zero `tqdm` progress behavior with total `config.max_train_steps`, resumed initial step, and a postfix for total, MSE, RGB MS-SSIM loss, depth SSI loss, point-map loss, learning rate, and gradient norm. Capture `grad_norm` from `optimizer_update(...)` and `learning_rate` from `lr_scheduler.get_last_lr()[0]` after every synchronized update. Reject a non-finite Stage-2 MSE with `RuntimeError("Stage-2 loss became non-finite")`, matching Stage 1's fail-fast safeguard.

On the main process only, call `accelerator.log(stage2_metric_values(...), step=step)` once per synchronized update. On the first such update after `_weights` returns—whether those weights were newly calibrated or loaded from `stage2_loss_calibration.json`—log `stage2_calibration_metric_values(weights)` once and record that it has been emitted. Do not change loss computation, calibration math, checkpoint format, or model architecture.

Mirror Stage 1's operator messages: print the resumed checkpoint and step, checkpoint path, and fixed-rollout artifact paths. After rank zero saves artifacts, call `accelerator.wait_for_everyone()` before further training.

- [ ] **Step 4: Verify GREEN**

Run: `PYTHONPATH=. pytest tests/training/test_stage2_smoke_cli.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/train_fixed_view_simgen_stage2.py tests/training/test_stage2_smoke_cli.py
git commit -m "feat: add stage 2 operational telemetry"
```

### Task 3: Mirror Stage-1 launcher diagnostics and document Stage-2 operations

**Files:**
- Modify: `submit_fixed_view_simgen_stage2_4gpu_mamba.sh:1-33`
- Modify: `tests/training/test_stage2_launcher.py`
- Modify: `docs/fixed_view_simgen_training.md`

**Interfaces:**
- Consumes Stage 2's unchanged `--preflight` and Python-module Accelerate command.
- Produces append-safe Slurm logs, an isolated Stage-2 NCCL diagnostic directory, early execution-environment failures, and operator guidance for W&B and Stage-2 metrics.

- [ ] **Step 1: Write failing launcher tests**

Extend `tests/training/test_stage2_launcher.py` to assert the Stage-2 launcher contains:

```python
assert "#SBATCH --ntasks-per-node=1" in launcher
assert "#SBATCH --open-mode=append" in launcher
assert 'NCCL_DEBUG_FILE="${PROJECT_DIR}/logs/nccl-aether-fixed-view-stage2-${SLURM_JOB_ID}/nccl.%h.%p.log"' in launcher
assert 'mkdir -p logs outputs/fixed_view_simgen_stage2' in launcher
assert '"${PYTHON_BIN}" -c' in launcher
assert "accelerate" in launcher and "wandb" in launcher
```

Keep the existing assertion that the final launch uses `"${PYTHON_BIN}" -m accelerate.commands.accelerate_cli launch`, preventing an accidental switch to a separate Accelerate executable. Do not change the Stage-1 launcher test or source.

- [ ] **Step 2: Verify RED**

Run: `PYTHONPATH=. pytest tests/training/test_stage2_launcher.py -v`

Expected: FAIL because the Stage-2 launcher does not yet request append mode, set NCCL diagnostics, create required directories, or verify its runtime environment.

- [ ] **Step 3: Implement the compatible operational baseline**

In the Stage-2 launcher:

1. Add `#SBATCH --ntasks-per-node=1` and `#SBATCH --open-mode=append`.
2. Create `logs`, `outputs/fixed_view_simgen_stage2`, and `logs/nccl-aether-fixed-view-stage2-${SLURM_JOB_ID}` after entering `PROJECT_DIR`.
3. Copy Stage 1's NCCL debug, interface/family, blocking-wait, and async-error-handling environment settings, changing only `NCCL_DEBUG_FILE` to the Stage-2-specific directory.
4. Verify that `${PYTHON_BIN}` is executable. Use it to import `torch`, `accelerate`, `diffusers`, `h5py`, `transformers`, `tiktoken`, `torchmetrics`, and `wandb`; print Torch/CUDA versions. Call `nvidia-smi` for driver/GPU diagnostics.
5. Print job ID, restart count, host, environment path, local model paths, Stage-1 source checkpoint, and start time.
6. Retain the existing Stage-2 `--preflight`; do not repeat its model/data/checkpoint/four-GPU validation in shell. Retain all Stage-2 override paths and the Python-module Accelerate invocation.

Update `docs/fixed_view_simgen_training.md` with a Stage-2 section covering submission, append/requeue diagnostics, the Stage-2 NCCL directory, W&B authentication (`wandb login` or `WANDB_API_KEY`), the `report_to=null` opt-out, and the exact Stage-2 scalar names. State that no media or artifact uploads occur.

- [ ] **Step 4: Verify GREEN and regressions**

Run:

```bash
PYTHONPATH=. pytest tests/training/test_stage2_launcher.py -v
bash -n submit_fixed_view_simgen_stage2_4gpu_mamba.sh
PYTHONPATH=. pytest tests/training -v
git diff --check
```

Expected: all tests pass, shell syntax is valid, and the diff has no whitespace errors. Do not submit a Slurm job or initialize a real W&B run.

- [ ] **Step 5: Commit**

```bash
git add submit_fixed_view_simgen_stage2_4gpu_mamba.sh \
  tests/training/test_stage2_launcher.py docs/fixed_view_simgen_training.md
git commit -m "feat: harden stage 2 launch operations"
```

## Self-Review

- **Spec coverage:** Tasks preserve the existing Stage-2 checkpoint/input contract while filling only the operational and tracking gaps. Task 1 defines W&B validation and payload contracts; Task 2 connects them to the Stage-2 lifecycle; Task 3 supplies the compatible Slurm baseline and operator documentation.
- **Step scan:** Every implementation step names exact files, interfaces, keys, commands, and unchanged boundaries. No task leaves a choice about metrics, launcher ownership, or the accelerator invocation.
- **Type consistency:** The tracker initializer returns `bool`; metric values are `dict[str, float]`; Task 2 consumes both as defined by Task 1.
- **Review focus:** Each listed failure mode has an owning task and a named automated test. External H200/W&B execution remains explicitly outside scope.
- **Proportion:** The plan describes interfaces and behavior rather than reproducing implementation bodies; only fixed metric/config values are included verbatim.
