# Fixed-View SimGen Stage-2 Training Design

## Purpose

Extend the completed fixed-view SimGen Stage-1 run with the image-space
refinement stage described in Aether, *Geometric-Aware Unified World
Modeling* (arXiv:2503.18945). The result is a separately resumable four-H200
job that starts from the completed Stage-1 transformer and improves RGB,
depth, and raymap outputs without modifying the Stage-1 artifacts.

## Scope and success criteria

This is a continuation of the existing fixed-view dynamics experiment, not an
implementation of Aether's full multi-task data mixture. It must:

- use the same 128 `sample_<id>/view_0` SimGen clips, 41 frames, 480x720
  padded tensors, four history slots, and four-H200 bf16 DDP topology as
  Stage 1;
- require the completed default Stage-1 checkpoint
  `${PROJECT_DIR}/outputs/fixed_view_simgen/checkpoint-010000` and fail before
  training if it is missing or incomplete;
- import *only* that checkpoint's transformer weights; it must not restore the
  Stage-1 optimizer, LR scheduler, RNG, or global step;
- write its own state below `outputs/fixed_view_simgen_stage2/`, so it can
  resume only its own checkpoints and never changes Stage-1 output;
- run 2,500 optimizer updates (one quarter of the 10,000 Stage-1 updates),
  checkpointing and producing deterministic fixed-sample artifacts every 500
  updates, including update 2,500; and
- fail clearly during preflight for an absent dependency, incompatible
  checkpoint, unavailable bf16 CUDA, or a GPU count other than four.

Stage 2 is successful when the Slurm launcher can preflight and launch this
path offline, stop/requeue/resume from its own checkpoints, and record all
losses and deterministic previews at the stated boundaries.

## Training state and scheduling

The Stage-2 trainer builds the same Aether input tensors as Stage 1. It loads
the frozen CogVideoX VAE and text encoder plus the Aether transformer, reads
the Stage-1 transformer state into the latter, and then constructs a new
AdamW optimizer. The VAE and text encoder remain parameter-frozen; gradients
must nevertheless flow through the VAE decoder to the predicted latent
inputs.

Stage 2 uses a fresh 2,500-update `OneCycleLR` schedule whose maximum learning
rate is `1e-5`. Its increasing phase lasts 250 updates (10% of the run). AdamW
uses the established fixed-view parameters: beta1 `0.9`, beta2 `0.95`, epsilon
`1e-8`, weight decay `0.1`, and maximum gradient norm `2.0`. The per-GPU batch
size and accumulation stay at one. A Stage-2 checkpoint restores this new
optimizer, scheduler, progress counter, calibrated auxiliary weights, and
run manifest; it does not revisit Stage-1 initialization.

The Stage-2 configuration is a separate strict config type. It must not relax
the Stage-1 config's `max_train_steps == 10000` contract.

## Objective

For each randomly selected diffusion timestep, the transformer predicts the
same epsilon or velocity target used by Stage 1. The base objective remains
the full latent-space diffusion MSE over all 56 target channels and all 11
latent frames.

The trainer converts the prediction and noisy latent back to an estimated
clean latent for either supported scheduler prediction type (`epsilon` or
`v_prediction`). It then decodes the RGB and disparity latent channel blocks
with the frozen VAE. Decoded losses apply to all 41 reconstructed video
frames, including observed history, because the model predicts the whole
video and the paper does not exclude conditioned frames.

The total objective is:

```text
diffusion_mse
+ lambda_rgb * rgb_ms_ssim
+ lambda_depth * depth_ssi
+ lambda_pointmap * raymap_pointmap
```

All loss reductions use only the dataset's `content_mask`: columns `[120,
600)` at all 480 rows. The artificial left/right padding must contribute no
decoded-loss gradient.

### RGB MS-SSIM

Map decoded RGB from the VAE range to `[0, 1]`, crop/mask to valid image
content, and minimize `1 - MS-SSIM` against the padded dataset RGB values.
Use the MS-SSIM implementation supplied by the existing `torchmetrics`
dependency, with `data_range=1.0`; no new perceptual-model download is
permitted.

### Depth SSI

Average the three decoded disparity channels, map from `[-1, 1]` to `[0, 1]`,
and square the result, matching Aether's existing decoded-disparity path.
Compare this normalized relative disparity to the square of the dataset's
`[-1, 1]` normalized-disparity target. For each sample, solve the least
squares scale and shift over valid content pixels, then use an L1 aligned-data
term plus the L1 horizontal and vertical gradient residuals. The two terms
have equal weight. Solves with a singular design matrix must use a
finite, epsilon-stabilized fallback.

This is an SSI loss on the representation actually available to the
fixed-view path. The data loader intentionally normalizes depth per clip and
does not retain metric scale, so it cannot supervise metric depth.

### Raymap pointmap loss

Unpack predicted and target 24-channel raymap latents into six-channel
per-frame raymaps, discard the four leading temporal pad frames, and align
them with the 41 decoded disparity frames. Bilinearly upsample ray directions
and signed-log ray origins by eight to decoded resolution. Construct each
pointmap as:

```text
pointmap = depth * ray_direction + ray_origin
```

where `depth` is reciprocal normalized disparity with a numerical lower
bound. Before comparison, align the predicted pointmap to the target with the
least-squares per-clip global scale and 3D translation allowed by the paper's
scale-and-translation-invariant description. Use a valid-content, inverse
target-depth-weighted L1 residual.

The pointmap calculation uses `decoded_disparity.detach()` for the predicted
depth factor. Thus this loss back-propagates into raymap latents only; the SSI
loss is the direct depth supervisor. This follows the paper's explicit
stop-gradient rule.

### Loss-weight calibration

The paper specifies the losses but not numeric coefficients. The Stage-2 YAML
exposes three positive auxiliary weights. At a fresh Stage-2 start, rank zero
computes the four raw losses from the first synchronized batch without an
optimizer update and derives multipliers that make each nonzero auxiliary
term initially equal in magnitude to the diffusion MSE. It stores the
effective multipliers in `stage2_loss_calibration.json`, broadcasts them to
all ranks, and includes them in Stage-2 checkpoint metadata. A restarted job
must reuse the stored values rather than recalibrating. The trainer logs raw
and weighted RGB, depth, and pointmap losses plus total loss.

## Checkpointing, artifacts, and launch contract

Stage-2 checkpoint metadata has a distinct `stage2_image_space_refinement`
objective identity and fingerprints the frozen models, dataset, loss contract,
and calibrated weights. A Stage-2 resume validates this identity before
calling `Accelerator.load_state`. It retains the newest two checkpoints.

At steps 500, 1000, 1500, 2000, and 2500, rank zero saves a Stage-2 checkpoint
and deterministic `sample_0` seed-42 target/generated RGB and disparity MP4s.
The preview remains a real 50-step Aether DPM rollout and has the observed
history composited exactly as Stage 1 does. Other ranks wait at save/preview
boundaries.

The new Slurm script mirrors the current H200/Mamba launcher and runs offline.
Before `accelerate launch`, its preflight must check the local Aether and
CogVideoX assets, Stage-1 checkpoint, the `tiktoken` import required by the
currently installed tokenizer, all image-space-loss imports, CUDA bf16, and
four visible GPUs. It passes explicit local model paths and the default
Stage-1 checkpoint location.

## Out of scope

- Changing Stage-1 code, checkpoints, hyperparameters, or output retention.
- Training the VAE or text encoder.
- Moving-camera data, full multi-task masking, planning, or action-dropout
  objectives.
- Metric-depth supervision, validation-set selection, or a new model/data
  download at launch time.
- Altering the original `submit_fixed_view_simgen_4gpu_mamba.sh` launcher.

## Verification requirements

Unit tests must cover clean-latent reconstruction for epsilon and velocity
prediction, valid-content masking, decoded-loss finite values and gradients,
depth-detached pointmap gradients, 41-frame raymap packing/unpacking,
calibration persistence/reuse, model-only Stage-1 initialization, and
Stage-2 checkpoint/resume isolation. Static tests must validate the new YAML
and Slurm launcher. A single-GPU smoke mode must execute one finite Stage-2
optimizer update; four-GPU preflight must verify the production layout without
starting optimization.
