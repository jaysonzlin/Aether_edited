"""Stage-1, fixed-view SimGen fine-tuning entry point for Aether."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from aether.training.config import load_training_config
from aether.training.simgen_dataset import SimGenFixedViewDataset


FIXED_VIEW_ROLLOUT_FPS = 12


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/train/fixed_view_simgen_4h200.yaml",
        help="Path to the fixed-view YAML configuration.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Portable config override in key=value form; may be repeated.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate all configured SimGen samples without importing model code.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a finite CPU Stage-1 MSE check without downloading weights.",
    )
    parser.add_argument(
        "--gpu-smoke-test",
        action="store_true",
        help="Load configured weights and data, run a real CUDA optimizer smoke test, then exit.",
    )
    parser.add_argument(
        "--gpu-smoke-test-steps",
        type=_positive_int,
        default=None,
        help="Optimizer updates for --gpu-smoke-test (default: 1).",
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Validate data, local model layout, and four-GPU bf16 readiness without training.",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Checkpoint directory to restore; defaults to latest for training and fresh state for GPU smoke tests.",
    )
    parser.add_argument(
        "--force-input-rehash",
        action="store_true",
        help="Recompute all model and dataset SHA-256 fingerprints instead of reusing the metadata cache.",
    )
    return parser.parse_args()


def _run_smoke_test() -> None:
    import torch
    import torch.nn.functional as functional

    target_noise = torch.randn(1, 11, 56, 2, 2)
    predicted_noise = torch.nn.Conv3d(56, 56, kernel_size=1)(
        target_noise.permute(0, 2, 1, 3, 4)
    ).permute(0, 2, 1, 3, 4)
    loss = functional.mse_loss(predicted_noise, target_noise)
    if not torch.isfinite(loss):
        raise RuntimeError("smoke-test Stage-1 MSE was non-finite")
    print(f"smoke-test Stage-1 MSE: {loss.item():.6f}")


def _load_training_components(config, accelerator):
    import torch
    from diffusers import (
        AutoencoderKLCogVideoX,
        CogVideoXDPMScheduler,
        CogVideoXTransformer3DModel,
    )
    from transformers import AutoTokenizer, T5EncoderModel

    from aether.pipelines.aetherv1_pipeline_cogvideox import AetherV1PipelineCogVideoX

    tokenizer = AutoTokenizer.from_pretrained(config.cogvideox_model_id, subfolder="tokenizer")
    text_encoder = T5EncoderModel.from_pretrained(
        config.cogvideox_model_id, subfolder="text_encoder", torch_dtype=torch.bfloat16
    )
    vae = AutoencoderKLCogVideoX.from_pretrained(
        config.cogvideox_model_id, subfolder="vae", torch_dtype=torch.bfloat16
    )
    scheduler = CogVideoXDPMScheduler.from_pretrained(
        config.cogvideox_model_id, subfolder="scheduler"
    )
    transformer = CogVideoXTransformer3DModel.from_pretrained(
        config.aether_model_id, subfolder="transformer", torch_dtype=torch.bfloat16
    )
    pipeline = AetherV1PipelineCogVideoX(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        scheduler=scheduler,
        transformer=transformer,
    )
    vae.requires_grad_(False).eval().to(accelerator.device)
    text_encoder.requires_grad_(False).eval()
    prompt_embeds = pipeline.empty_prompt_embeds.to(accelerator.device)
    return pipeline, transformer, vae, scheduler, prompt_embeds


def _infinite_batches(dataloader):
    while True:
        yield from dataloader


def training_steps(
    max_train_steps: int,
    gpu_smoke_test: bool,
    gpu_smoke_test_steps: int = 1,
) -> int:
    """Keep the saved run configuration intact while capping GPU smoke mode."""
    if not gpu_smoke_test:
        return max_train_steps
    if gpu_smoke_test_steps <= 0:
        raise ValueError("gpu_smoke_test_steps must be positive")
    return min(gpu_smoke_test_steps, max_train_steps)


def validate_gpu_smoke_test_args(args: argparse.Namespace) -> None:
    if args.gpu_smoke_test_steps is not None and not args.gpu_smoke_test:
        raise SystemExit("--gpu-smoke-test-steps requires --gpu-smoke-test")
    if args.gpu_smoke_test and args.resume is not None:
        raise SystemExit("--gpu-smoke-test cannot be combined with --resume")


def diffusion_training_target(scheduler, target_latents, noise, timesteps):
    """Return the Stage-1 target matching the loaded scheduler parameterization."""
    prediction_type = getattr(scheduler.config, "prediction_type", None)
    if prediction_type == "epsilon":
        return noise
    if prediction_type == "v_prediction":
        return scheduler.get_velocity(target_latents, noise, timesteps)
    raise ValueError(
        "unsupported scheduler prediction_type "
        f"{prediction_type!r}; expected 'epsilon' or 'v_prediction'"
    )


def component_training_losses(prediction, target):
    """Report detached per-element MSEs for Aether's RGB-D and raymap outputs."""
    import torch.nn.functional as functional

    from aether.training.aether_latents import RGB_LATENT_CHANNELS, RAYMAP_CHANNELS

    if prediction.shape != target.shape or prediction.ndim != 5:
        raise ValueError("prediction and target must have matching [batch, frames, channels, height, width] shapes")
    disparity_start = RGB_LATENT_CHANNELS
    raymap_start = RGB_LATENT_CHANNELS * 2
    expected_channels = raymap_start + RAYMAP_CHANNELS
    if prediction.shape[2] != expected_channels:
        raise ValueError(f"expected {expected_channels} output channels, got {prediction.shape[2]}")

    prediction = prediction.detach().float()
    target = target.detach().float()
    return {
        "train/rgb_loss": functional.mse_loss(
            prediction[:, :, :disparity_start], target[:, :, :disparity_start]
        ),
        "train/disparity_loss": functional.mse_loss(
            prediction[:, :, disparity_start:raymap_start],
            target[:, :, disparity_start:raymap_start],
        ),
        "train/raymap_loss": functional.mse_loss(
            prediction[:, :, raymap_start:], target[:, :, raymap_start:]
        ),
    }


def build_optimizer(transformer, config, torch_module=None):
    """Build the Wan-video AdamW variant used by this fine-tuning path."""
    if torch_module is None:
        import torch as torch_module
    return torch_module.optim.AdamW(
        transformer.parameters(),
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_epsilon,
        weight_decay=config.weight_decay,
    )


def build_lr_scheduler(optimizer, config):
    from transformers import get_constant_schedule_with_warmup

    return get_constant_schedule_with_warmup(
        optimizer, num_warmup_steps=config.warmup_steps
    )


def accelerator_options(config) -> dict[str, object]:
    """Keep prepared LR schedulers on optimizer-update rather than per-rank time."""
    return {
        "mixed_precision": config.mixed_precision,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "log_with": config.report_to,
        "step_scheduler_with_optimizer": False,
    }


def optimizer_update(accelerator, optimizer, lr_scheduler, transformer, max_grad_norm):
    """Clip before synchronized updates and advance warmup only per optimizer step."""
    grad_norm = None
    if accelerator.sync_gradients:
        grad_norm = accelerator.clip_grad_norm_(
            transformer.parameters(), max_grad_norm
        )
    optimizer.step()
    if accelerator.sync_gradients:
        lr_scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    return grad_norm


def has_remaining_steps(completed_steps: int, max_train_steps: int) -> bool:
    return completed_steps < max_train_steps


def advance_optimizer_step(global_step: int, sync_gradients: bool) -> int:
    """Count optimizer updates, not the microbatches used to accumulate them."""
    return global_step + int(sync_gradients)


def optimizer_step_status(
    global_step: int,
    sync_gradients: bool,
    max_train_steps: int,
    output_interval: int,
) -> tuple[int, bool, bool]:
    """Return updated step, checkpoint boundary, and completion after a microbatch."""
    if not sync_gradients:
        return global_step, False, False
    global_step = advance_optimizer_step(global_step, True)
    return (
        global_step,
        global_step % output_interval == 0,
        global_step >= max_train_steps,
    )


def resolve_resume_checkpoint(output_dir, resume: str | None, gpu_smoke_test: bool):
    """Resolve normal resume defaults while guaranteeing a fresh smoke test."""
    from aether.training.checkpointing import latest_checkpoint

    if gpu_smoke_test:
        if resume is not None:
            raise ValueError("--gpu-smoke-test cannot be combined with --resume")
        return None
    if resume in (None, "latest"):
        return latest_checkpoint(output_dir)
    return Path(resume)


def shared_run_manifest(
    config, prediction_type, accelerator, torch_module, force_rehash: bool = False
):
    """Fingerprint large assets once, then share the manifest with all DDP ranks."""
    from aether.training.checkpointing import build_run_manifest

    manifest = (
        build_run_manifest(
            config,
            prediction_type,
            show_progress=True,
            fingerprint_cache_path=Path(config.output_dir)
            / "input_fingerprint_cache.json",
            force_rehash=force_rehash,
        )
        if accelerator.is_main_process
        else None
    )
    manifests = [manifest]
    if accelerator.num_processes > 1:
        torch_module.distributed.broadcast_object_list(manifests, src=0)
    if manifests[0] is None:
        raise RuntimeError("rank zero did not broadcast the checkpoint run manifest")
    return manifests[0]


def unwrapped_transformer_config(transformer, accelerator):
    """Read architecture settings from the model behind an Accelerate wrapper."""
    return accelerator.unwrap_model(transformer).config


def run_preflight(config, torch_module=None) -> None:
    """Reject a cluster allocation that cannot run the fixed four-H200 job."""
    if torch_module is None:
        import torch as torch_module

    validate_local_model_files(config)
    SimGenFixedViewDataset(config.data_root, config.sample_ids)
    if not torch_module.cuda.is_available():
        raise RuntimeError("preflight requires CUDA")
    if torch_module.cuda.device_count() != 4:
        raise RuntimeError("preflight requires exactly four visible GPUs")
    if not torch_module.cuda.is_bf16_supported():
        raise RuntimeError("preflight requires CUDA bf16 support")
    print("preflight passed: data, local model layout, four GPUs, and bf16 are ready")


def validate_local_model_files(config) -> None:
    """Fail before model loading when local Hugging Face components are incomplete."""
    aether_transformer = Path(config.aether_model_id) / "transformer"
    cogvideox_root = Path(config.cogvideox_model_id)
    missing = []

    def require_file(path: Path) -> None:
        if not path.is_file():
            missing.append(str(path))

    def require_weights(component: Path) -> None:
        patterns = ("*.safetensors", "*.bin", "*.pt", "*.pth")
        if not any(any(component.rglob(pattern)) for pattern in patterns):
            missing.append(f"{component}/<model weights (*.safetensors or *.bin)>")

    require_file(aether_transformer / "config.json")
    require_weights(aether_transformer)

    tokenizer = cogvideox_root / "tokenizer"
    require_file(tokenizer / "tokenizer_config.json")
    tokenizer_assets = (
        tokenizer / "tokenizer.json",
        tokenizer / "spiece.model",
        tokenizer / "vocab.json",
    )
    if not any(path.is_file() for path in tokenizer_assets):
        missing.append(f"{tokenizer}/<tokenizer.json, spiece.model, or vocab.json>")

    for component_name in ("text_encoder", "vae"):
        component = cogvideox_root / component_name
        require_file(component / "config.json")
        require_weights(component)
    require_file(cogvideox_root / "scheduler" / "scheduler_config.json")

    if missing:
        raise FileNotFoundError(
            "incomplete local model files; missing required artifacts: "
            + ", ".join(missing)
        )


def fixed_rollout_batch(dataset, device):
    """Load the unshuffled fixed sample used for every qualitative preview."""
    import torch

    try:
        sample_index = tuple(dataset.sample_ids).index(0)
    except ValueError as error:
        raise ValueError("fixed rollout requires sample_0 in the training dataset") from error
    sample = dataset[sample_index]
    return {
        key: torch.as_tensor(sample[key], device=device).unsqueeze(0)
        for key in ("rgb", "disparity", "raymap")
    }


def _decoded_rgb_batch(decoded_video):
    """Normalize CogVideoX postprocessed video to [batch, frames, channels, H, W]."""
    import torch

    decoded = torch.as_tensor(decoded_video)
    if decoded.ndim != 5 or decoded.shape[-1] != 3:
        raise ValueError("CogVideoX decode must have shape [batch, frames, height, width, 3]")
    return decoded.permute(0, 1, 4, 2, 3)


def save_fixed_rollout_artifacts(
    *,
    accelerator,
    config,
    dataset,
    pipeline,
    transformer,
    vae,
    scheduler,
    prompt_embeds,
    global_step: int,
) -> tuple[Path, Path, Path, Path]:
    """Generate rank-zero's deterministic fixed-sample target and rollout MP4s."""
    import torch

    from aether.training.aether_latents import assemble_aether_training_batch
    from aether.training.rollout import (
        composite_history,
        require_41_frames,
        sample_aether_latents,
    )
    from aether.training.visualization import (
        decoded_disparity_to_viridis_frames,
        normalized_disparity_to_viridis_frames,
        save_fixed_rollout,
    )

    fixed_batch = fixed_rollout_batch(dataset, accelerator.device)
    cuda_devices = [accelerator.device.index or 0] if accelerator.device.type == "cuda" else []
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(config.seed)
        if accelerator.device.type == "cuda":
            torch.cuda.manual_seed(config.seed)
        assembled = assemble_aether_training_batch(fixed_batch, vae, config.history_slots)

    rollout_model = accelerator.unwrap_model(transformer)
    was_training = rollout_model.training
    rollout_model.eval()
    try:
        rotary_emb = (
            pipeline._prepare_rotary_positional_embeddings(
                config.height,
                config.width,
                assembled.target_latents.shape[1],
                accelerator.device,
                fps=FIXED_VIEW_ROLLOUT_FPS,
            )
            if rollout_model.config.use_rotary_positional_embeddings
            else None
        )
        ofs = (
            None
            if rollout_model.config.ofs_embed_dim is None
            else assembled.target_latents.new_full((1,), 2.0)
        )
        rollout_latents = sample_aether_latents(
            rollout_model,
            scheduler,
            assembled.condition_latents,
            prompt_embeds,
            rotary_emb,
            ofs,
            seed=config.seed,
            num_inference_steps=50,
            show_progress=accelerator.is_main_process,
        )
        rgb_latents = rollout_latents[:, :, :16]
        disparity_latents = rollout_latents[:, :, 16:32]
        decoded_rgb = pipeline.video_processor.postprocess_video(
            video=pipeline.decode_latents(rgb_latents), output_type="np"
        )
        decoded_disparity = pipeline.decode_latents(disparity_latents)
        generated_rgb = _decoded_rgb_batch(decoded_rgb).to(fixed_batch["rgb"].device)
        require_41_frames(generated_rgb)
        generated_rgb = composite_history(generated_rgb, fixed_batch["rgb"], history_frames=13)
        generated_disparity = decoded_disparity_to_viridis_frames(
            decoded_disparity.detach().float().cpu().numpy()
        )
        target_disparity = normalized_disparity_to_viridis_frames(
            fixed_batch["disparity"][0].detach().float().cpu().numpy()
        )
        return save_fixed_rollout(
            config.output_dir,
            global_step,
            target_rgb=fixed_batch["rgb"][0].permute(0, 2, 3, 1).cpu().numpy(),
            generated_rgb=generated_rgb[0].permute(0, 2, 3, 1).cpu().numpy(),
            target_disparity=target_disparity,
            generated_disparity=generated_disparity,
            fps=FIXED_VIEW_ROLLOUT_FPS,
        )
    finally:
        rollout_model.train(was_training)


def run_training(
    config,
    gpu_smoke_test: bool = False,
    resume: str | None = None,
    gpu_smoke_test_steps: int = 1,
    force_input_rehash: bool = False,
) -> None:
    import torch
    import torch.nn.functional as functional
    from accelerate import Accelerator
    from tqdm.auto import tqdm
    from torch.utils.data import DataLoader

    from aether.training.aether_latents import assemble_aether_training_batch
    from aether.training.checkpointing import (
        restore_checkpoint,
        save_checkpoint,
    )

    accelerator = Accelerator(**accelerator_options(config))
    if config.report_to:
        accelerator.init_trackers(
            config.wandb_project,
            config={
                "learning_rate": config.learning_rate,
                "max_train_steps": config.max_train_steps,
                "warmup_steps": config.warmup_steps,
                "adam_beta1": config.adam_beta1,
                "adam_beta2": config.adam_beta2,
                "adam_epsilon": config.adam_epsilon,
                "weight_decay": config.weight_decay,
                "max_grad_norm": config.max_grad_norm,
            },
        )
    if gpu_smoke_test and (not torch.cuda.is_available() or accelerator.device.type != "cuda"):
        raise RuntimeError("--gpu-smoke-test requires an available CUDA GPU")
    torch.manual_seed(config.seed)
    max_train_steps = training_steps(
        config.max_train_steps, gpu_smoke_test, gpu_smoke_test_steps
    )
    dataset = SimGenFixedViewDataset(config.data_root, config.sample_ids)
    dataloader = DataLoader(
        dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=config.dataloader_num_workers,
        pin_memory=config.pin_memory,
    )
    pipeline, transformer, vae, scheduler, prompt_embeds = _load_training_components(
        config, accelerator
    )
    run_manifest = shared_run_manifest(
        config,
        scheduler.config.prediction_type,
        accelerator,
        torch,
        force_rehash=force_input_rehash,
    )
    if hasattr(transformer, "enable_gradient_checkpointing"):
        transformer.enable_gradient_checkpointing()
    optimizer = build_optimizer(transformer, config, torch)
    lr_scheduler = build_lr_scheduler(optimizer, config)
    transformer, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, dataloader, lr_scheduler
    )
    transformer_config = unwrapped_transformer_config(transformer, accelerator)
    transformer.train()
    checkpoint = resolve_resume_checkpoint(config.output_dir, resume, gpu_smoke_test)
    completed_steps = 0
    if checkpoint is not None:
        completed_steps = restore_checkpoint(
            accelerator, checkpoint, expected_manifest=run_manifest
        )
        accelerator.print(f"resumed checkpoint {checkpoint} at step {completed_steps}")
    if not has_remaining_steps(completed_steps, max_train_steps):
        accelerator.print(f"training already completed at step {completed_steps}")
        if config.report_to:
            accelerator.end_training()
        return

    global_step = completed_steps
    progress = tqdm(
        total=max_train_steps,
        initial=completed_steps,
        desc="Aether training",
        unit="step",
        disable=not accelerator.is_main_process,
    )
    try:
        for batch in _infinite_batches(dataloader):
            with accelerator.accumulate(transformer):
                model_batch = {
                    key: value.to(accelerator.device)
                    for key, value in batch.items()
                    if key in {"rgb", "disparity", "raymap"}
                }
                assembled = assemble_aether_training_batch(model_batch, vae, config.history_slots)
                target_latents = assembled.target_latents
                noise = torch.randn_like(target_latents)
                timesteps = torch.randint(
                    0,
                    scheduler.config.num_train_timesteps,
                    (target_latents.shape[0],),
                    device=target_latents.device,
                ).long()
                noisy_latents = scheduler.add_noise(target_latents, noise, timesteps)
                rotary_emb = (
                    pipeline._prepare_rotary_positional_embeddings(
                        config.height,
                        config.width,
                        noisy_latents.shape[1],
                        accelerator.device,
                        fps=12,
                    )
                    if transformer_config.use_rotary_positional_embeddings
                    else None
                )
                ofs = (
                    None
                    if transformer_config.ofs_embed_dim is None
                    else noisy_latents.new_full((1,), 2.0)
                )
                prediction = transformer(
                    hidden_states=torch.cat((noisy_latents, assembled.condition_latents), dim=2),
                    encoder_hidden_states=prompt_embeds.repeat(target_latents.shape[0], 1, 1),
                    timestep=timesteps,
                    ofs=ofs,
                    image_rotary_emb=rotary_emb,
                    return_dict=False,
                )[0]
                training_target = diffusion_training_target(
                    scheduler, target_latents, noise, timesteps
                )
                loss = functional.mse_loss(prediction.float(), training_target.float())
                component_losses = component_training_losses(prediction, training_target)
                if not torch.isfinite(loss):
                    raise RuntimeError("Stage-1 loss became non-finite")
                accelerator.backward(loss)
                grad_norm = optimizer_update(
                    accelerator,
                    optimizer,
                    lr_scheduler,
                    transformer,
                    config.max_grad_norm,
                )

            global_step, checkpoint_due, training_complete = optimizer_step_status(
                global_step,
                accelerator.sync_gradients,
                max_train_steps,
                config.output_interval,
            )
            if accelerator.sync_gradients:
                learning_rate = lr_scheduler.get_last_lr()[0]
                grad_norm_value = (
                    float(grad_norm.detach().float().item())
                    if grad_norm is not None
                    else 0.0
                )
                component_loss_values = {
                    name: float(value.item()) for name, value in component_losses.items()
                }
                if accelerator.is_main_process:
                    progress.update(1)
                    progress.set_postfix(
                        loss=f"{loss.item():.5f}",
                        rgb=f"{component_loss_values['train/rgb_loss']:.5f}",
                        disparity=f"{component_loss_values['train/disparity_loss']:.5f}",
                        raymap=f"{component_loss_values['train/raymap_loss']:.5f}",
                        lr=f"{learning_rate:.2e}",
                        grad_norm=f"{grad_norm_value:.3f}",
                    )
                if config.report_to and accelerator.is_main_process:
                    accelerator.log(
                        {
                            "train/loss": float(loss.detach().float().item()),
                        "train/learning_rate": float(learning_rate),
                        "train/grad_norm": grad_norm_value,
                        **component_loss_values,
                        },
                        step=global_step,
                    )
                if checkpoint_due:
                    checkpoint = save_checkpoint(
                        accelerator,
                        config.output_dir,
                        global_step,
                        run_manifest=run_manifest,
                        keep_last=2,
                    )
                    if accelerator.is_main_process:
                        accelerator.print(f"saved checkpoint: {checkpoint}")
                        artifact_paths = save_fixed_rollout_artifacts(
                            accelerator=accelerator,
                            config=config,
                            dataset=dataset,
                            pipeline=pipeline,
                            transformer=transformer,
                            vae=vae,
                            scheduler=scheduler,
                            prompt_embeds=prompt_embeds,
                            global_step=global_step,
                        )
                        accelerator.print(
                            "saved fixed rollout artifacts: "
                            + ", ".join(str(path) for path in artifact_paths)
                        )
                    accelerator.wait_for_everyone()
                if training_complete:
                    break
    finally:
        progress.close()
        if config.report_to:
            accelerator.end_training()
    if gpu_smoke_test:
        accelerator.print(
            "GPU smoke test passed: "
            f"{max_train_steps} finite Stage-1 optimizer update(s) completed"
        )


def main() -> None:
    args = parse_args()
    validate_gpu_smoke_test_args(args)
    config = load_training_config(args.config, args.override)
    if args.dry_run:
        dataset = SimGenFixedViewDataset(config.data_root, config.sample_ids)
        print(f"validated {len(dataset)} fixed-view SimGen samples")
        return
    if args.preflight:
        run_preflight(config)
        return
    if args.smoke_test:
        _run_smoke_test()
        return
    run_training(
        config,
        gpu_smoke_test=args.gpu_smoke_test,
        resume=args.resume,
        gpu_smoke_test_steps=args.gpu_smoke_test_steps or 1,
        force_input_rehash=args.force_input_rehash,
    )


if __name__ == "__main__":
    main()
