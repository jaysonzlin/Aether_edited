# Fixed-View SimGen Aether Training Design

## Goal

Add a training path that full-fine-tunes the released Aether transformer on 128 fixed-camera SimGen RGB-D clips. The first experiment is an intentional overfit run for stable dynamics prediction, not a generalization benchmark.

## Scope

- Train only the Aether `CogVideoXTransformer3DModel` weights.
- Keep the CogVideoX VAE and text encoder frozen and use an empty text prompt.
- Use RGB, normalized-disparity, and stationary-raymap latent targets.
- Use 41 contiguous source frames at 12 fps, frames `00000000.png` through `00000040.png`.
- Pad each 480x480 RGB and depth frame to 480x720 with 120 pixels on each horizontal side; update `cx` by +120 before raymap generation.
- Use four Aether latent-time slots as RGB history and seven as future generation slots.
- Run Stage 1 latent diffusion MSE only. No decoded-image losses, point-cloud predictor, validation split, or action-condition dropout are in scope.
- Run 10,000 optimizer steps on one four-H200 node with bf16, gradient checkpointing, and FSDP/ZeRO-style sharding. Emit fixed-seed visualizations and checkpoints every 1,000 steps.

## Data Contract

Each `sample_<id>` beneath the supplied root contains:

- `view_0/00000000.png` through at least `view_0/00000040.png`, RGB 480x480 frames.
- `view_0/depth.h5`, whose `depth` dataset has per-frame depth aligned to the RGB frames.
- `view_0/cameras.json`, one stationary camera record per frame with `rotation`, `position`, `fx`, `fy`, `width`, and `height`.

The dataset validates every input before the first training batch. It returns RGB clips, normalized three-channel disparity clips, stationary padded raymaps, and an RGB valid-content mask for decoded visualizations. It must fail with a precise path-specific error for missing frames, malformed depth, a camera-count mismatch, unsupported dimensions, or non-finite valid depth.

The padding policy is part of the manifest: RGB uses edge replication, depth uses the per-clip maximum valid depth, and the original 480-pixel content occupies columns `[120, 600)`. The padding is deterministic and its geometric rays come from the widened intrinsic matrix; it is not a resize.

## Model Contract

The training module loads the released `AetherWorldModel/AetherV1` transformer and the compatible `THUDM/CogVideoX-5b-I2V` tokenizer, text encoder, VAE, and scheduler.

For each clip, it creates the 56-channel target latent state used by Aether: 16 RGB VAE channels, 16 disparity VAE channels, and 24 packed raymap channels. The condition state comprises 16 RGB condition channels plus 24 packed raymap condition channels. The first four RGB condition time positions contain the encoded history prefix and the remaining positions are zero. Raymap conditions are present for every position.

The prediction path retains Aether's separate condition tensor rather than importing Wan's point-cloud bridge or its latent-pinning implementation. The loss is the standard per-element noise-prediction MSE over the 56 target channels. A fixed history prefix is supplied at inference and is composited into saved RGB outputs for inspection.

## Operational Contract

- Configuration is YAML and contains no cluster-specific absolute paths.
- A four-GPU launch uses Accelerate/FSDP configuration supplied outside the code repository.
- Resume restores model, optimizer, scheduler, RNG, and global step from a numbered checkpoint.
- The trainer saves a target clip and a fixed-seed generated rollout at steps 1,000 through 10,000.
- Initial stability is measured only by finite, non-divergent loss and qualitative history/future continuity. Validation is deliberately deferred.

## Non-goals

- No training of the VAE or text encoder.
- No 49-frame Aether pipeline support.
- No moving cameras, goal planning, reconstruction task mixing, action dropout, PC prediction, or Stage 2 image-space losses.
