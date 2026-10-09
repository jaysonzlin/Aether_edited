"""Stage-2 image-space refinement for completed fixed-view SimGen Stage-1."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train/fixed_view_simgen_stage2_4h200.yaml")
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--resume", nargs="?", const="latest")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--gpu-smoke-test", action="store_true")
    parser.add_argument("--gpu-smoke-test-steps", type=_positive_int)
    parser.add_argument("--trace-smoke-test", action="store_true")
    args = parser.parse_args(argv)
    if args.gpu_smoke_test_steps is not None and not args.gpu_smoke_test:
        parser.error("--gpu-smoke-test-steps requires --gpu-smoke-test")
    return args


def _calibration_path(config):
    return Path(config.output_dir) / "stage2_loss_calibration.json"


def smoke_result_message(step: int, loss: float) -> str:
    return f"Stage-2 GPU smoke test passed: completed {step} optimizer update (loss={loss:.6f})"


def smoke_trace_message(boundary: str) -> str:
    return f"Stage-2 GPU smoke trace: {boundary}"


def enable_transformer_gradient_checkpointing(transformer) -> None:
    """Match Stage-1 activation checkpointing for 41-frame H200 training."""
    enable = getattr(transformer, "enable_gradient_checkpointing", None)
    if callable(enable):
        enable()


def enable_vae_gradient_checkpointing(vae) -> None:
    """Checkpoint frozen VAE decoder activations while retaining image-loss gradients."""
    enable = getattr(vae, "enable_gradient_checkpointing", None)
    if callable(enable):
        enable()


def stage2_tracker_config(config) -> dict[str, object]:
    return {
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


def initialize_stage2_tracking(accelerator, config) -> bool:
    if not config.report_to:
        return False
    accelerator.init_trackers(config.wandb_project, config=stage2_tracker_config(config))
    return True


def finish_stage2_tracking(accelerator, initialized: bool) -> None:
    if initialized:
        accelerator.end_training()


def _metric_float(value) -> float:
    detach = getattr(value, "detach", None)
    if callable(detach):
        return float(detach().float().item())
    return float(value)


def stage2_metric_values(*, total, mse, losses, learning_rate, grad_norm) -> dict[str, float]:
    return {
        "train/loss": _metric_float(total),
        "train/mse": _metric_float(mse),
        "train/rgb_ms_ssim": _metric_float(losses.rgb),
        "train/depth_ssi": _metric_float(losses.depth),
        "train/pointmap": _metric_float(losses.pointmap),
        "train/learning_rate": _metric_float(learning_rate),
        "train/grad_norm": _metric_float(grad_norm),
    }


def stage2_calibration_metric_values(weights) -> dict[str, float]:
    return {
        "train/calibrated_rgb_weight": _metric_float(weights["rgb"]),
        "train/calibrated_depth_weight": _metric_float(weights["depth"]),
        "train/calibrated_pointmap_weight": _metric_float(weights["pointmap"]),
    }


def _weights(config, mse, losses, accelerator):
    from aether.training.stage2_losses import calibrate_auxiliary_weights

    path = _calibration_path(config)
    if accelerator.is_main_process:
        if path.is_file():
            values = json.loads(path.read_text())
        else:
            values = calibrate_auxiliary_weights(mse, {"rgb": losses.rgb, "depth": losses.depth, "pointmap": losses.pointmap}, {"rgb": config.rgb_loss_weight, "depth": config.depth_loss_weight, "pointmap": config.pointmap_loss_weight})
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(values, sort_keys=True) + "\n")
    else:
        values = None
    values = [values]
    if accelerator.num_processes > 1:
        import torch
        torch.distributed.broadcast_object_list(values, src=0)
    return values[0]


def preflight(config):
    import torch
    import tiktoken  # noqa: F401
    import torchmetrics  # noqa: F401
    from scripts.train_fixed_view_simgen import validate_local_model_files
    from aether.training.simgen_dataset import FixedViewSimGenDataset

    validate_local_model_files(config)
    FixedViewSimGenDataset(config.data_root, config.sample_ids)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 4 or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Stage-2 preflight requires exactly four CUDA bf16 GPUs")
    checkpoint = Path(config.stage1_checkpoint)
    metadata = json.loads((checkpoint / "metadata.json").read_text())
    if metadata.get("global_step") != 10_000 or not (checkpoint / "model.safetensors").is_file():
        raise ValueError("Stage-2 requires a complete Stage-1 checkpoint-010000")
    print("Stage-2 preflight passed")


def main():
    args = parse_args()
    trace_smoke = args.gpu_smoke_test and args.trace_smoke_test
    step_limit = (args.gpu_smoke_test_steps or 1) if args.gpu_smoke_test else None

    def trace(boundary: str) -> None:
        if trace_smoke:
            print(smoke_trace_message(boundary), flush=True)

    trace("arguments parsed")
    from accelerate import Accelerator
    from torch.utils.data import DataLoader
    import torch
    import torch.nn.functional as functional
    from aether.training.aether_latents import assemble_aether_training_batch
    from aether.training.checkpointing import latest_checkpoint, load_stage1_transformer_weights, restore_checkpoint, save_checkpoint
    from aether.training.simgen_dataset import FixedViewSimGenDataset
    from aether.training.stage2_config import load_stage2_training_config
    from aether.training.stage2_losses import (
        latent_gradient_surrogate,
        measure_stage2_losses,
        reconstruct_clean_latents,
        stage2_latent_gradient,
    )
    from scripts.train_fixed_view_simgen import _load_training_components, accelerator_options, diffusion_training_target, optimizer_update, save_fixed_rollout_artifacts, unwrapped_transformer_config

    config = load_stage2_training_config(args.config, args.override)
    trace("configuration loaded")
    if args.preflight:
        preflight(config)
        return
    accelerator = Accelerator(**accelerator_options(config))
    tracking_initialized = initialize_stage2_tracking(accelerator, config)
    trace("accelerator created")
    dataset = FixedViewSimGenDataset(config.data_root, config.sample_ids)
    dataloader = DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=config.dataloader_num_workers, pin_memory=config.pin_memory)
    trace("dataset and dataloader created")
    pipeline, transformer, vae, scheduler, prompts = _load_training_components(config, accelerator)
    trace("components loaded")
    enable_transformer_gradient_checkpointing(transformer)
    trace("gradient checkpointing enabled")
    enable_vae_gradient_checkpointing(vae)
    trace("VAE gradient checkpointing enabled")
    load_stage1_transformer_weights(transformer, config.stage1_checkpoint)
    trace("stage-1 transformer weights loaded")
    optimizer = torch.optim.AdamW(transformer.parameters(), lr=config.learning_rate, betas=(config.adam_beta1, config.adam_beta2), eps=config.adam_epsilon, weight_decay=config.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config.learning_rate, total_steps=config.max_train_steps, pct_start=config.onecycle_pct_start)
    transformer, optimizer, dataloader, lr_scheduler = accelerator.prepare(transformer, optimizer, dataloader, lr_scheduler)
    trace("training state prepared")
    transformer_config = unwrapped_transformer_config(transformer, accelerator)
    manifest = {"schema_version": 1, "objective": "stage2_image_space_refinement", "stage1_checkpoint": str(config.stage1_checkpoint)}
    checkpoint = None if args.gpu_smoke_test else (latest_checkpoint(config.output_dir) if args.resume in (None, "latest") else Path(args.resume))
    step = restore_checkpoint(accelerator, checkpoint, manifest) if checkpoint else 0
    if checkpoint:
        accelerator.print(f"resumed checkpoint {checkpoint} at step {step}")
    weights = None
    calibration_logged = False
    while step < (step_limit or config.max_train_steps):
        for batch in dataloader:
            trace("first batch received")
            with accelerator.accumulate(transformer):
                batch = {key: value.to(accelerator.device) for key, value in batch.items() if key in {"rgb", "disparity", "raymap"}}
                assembled = assemble_aether_training_batch(batch, vae, config.history_slots)
                noise = torch.randn_like(assembled.target_latents)
                timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (noise.shape[0],), device=noise.device).long()
                noisy = scheduler.add_noise(assembled.target_latents, noise, timesteps)
                rotary = pipeline._prepare_rotary_positional_embeddings(config.height, config.width, noisy.shape[1], accelerator.device, fps=12) if transformer_config.use_rotary_positional_embeddings else None
                ofs = None if transformer_config.ofs_embed_dim is None else noisy.new_full((1,), 2.0)
                prediction = transformer(hidden_states=torch.cat((noisy, assembled.condition_latents), dim=2), encoder_hidden_states=prompts.repeat(noise.shape[0], 1, 1), timestep=timesteps, ofs=ofs, image_rotary_emb=rotary, return_dict=False)[0]
                target = diffusion_training_target(scheduler, assembled.target_latents, noise, timesteps)
                mse = functional.mse_loss(prediction.float(), target.float())
                if not torch.isfinite(mse):
                    raise RuntimeError("Stage-2 loss became non-finite")
                clean_latents = reconstruct_clean_latents(scheduler, noisy, prediction, timesteps)
                if weights is None:
                    calibration_losses = measure_stage2_losses(vae, clean_latents, batch)
                    weights = _weights(config, mse, calibration_losses, accelerator)
                losses, auxiliary_gradient = stage2_latent_gradient(
                    vae, clean_latents, batch, weights
                )
                total = mse.detach() + weights["rgb"] * losses.rgb + weights["depth"] * losses.depth + weights["pointmap"] * losses.pointmap
                accelerator.backward(mse + latent_gradient_surrogate(clean_latents, auxiliary_gradient))
                grad_norm = optimizer_update(accelerator, optimizer, lr_scheduler, transformer, config.max_grad_norm)
            if accelerator.sync_gradients:
                step += 1
                trace("optimizer update completed")
                learning_rate = lr_scheduler.get_last_lr()[0]
                if tracking_initialized and accelerator.is_main_process:
                    if not calibration_logged:
                        accelerator.log(stage2_calibration_metric_values(weights), step=step)
                        calibration_logged = True
                    accelerator.log(stage2_metric_values(total=total, mse=mse, losses=losses, learning_rate=learning_rate, grad_norm=0.0 if grad_norm is None else grad_norm), step=step)
                if step % config.output_interval == 0 or step == config.max_train_steps:
                    saved_checkpoint = save_checkpoint(accelerator, config.output_dir, step, manifest, keep_last=2)
                    accelerator.print(f"saved checkpoint: {saved_checkpoint}")
                    if accelerator.is_main_process:
                        artifacts = save_fixed_rollout_artifacts(accelerator, config, dataset, pipeline, transformer, vae, scheduler, prompts, step)
                        accelerator.print("saved fixed rollout artifacts: " + ", ".join(str(path) for path in artifacts))
                    accelerator.wait_for_everyone()
                if step >= (step_limit or config.max_train_steps):
                    if args.gpu_smoke_test:
                        print(
                            smoke_result_message(step, float(total.detach().float().item())),
                            flush=True,
                        )
                    finish_stage2_tracking(accelerator, tracking_initialized)
                    return


if __name__ == "__main__":
    if "--trace-smoke-test" in sys.argv:
        print(smoke_trace_message("entrypoint reached"), flush=True)
    main()
