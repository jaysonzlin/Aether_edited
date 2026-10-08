"""Stage-2 image-space refinement for completed fixed-view SimGen Stage-1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train/fixed_view_simgen_stage2_4h200.yaml")
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--resume", nargs="?", const="latest")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--gpu-smoke-test", action="store_true")
    return parser.parse_args()


def _calibration_path(config):
    return Path(config.output_dir) / "stage2_loss_calibration.json"


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
    from accelerate import Accelerator
    from torch.utils.data import DataLoader
    import torch
    import torch.nn.functional as functional
    from aether.training.aether_latents import assemble_aether_training_batch
    from aether.training.checkpointing import latest_checkpoint, load_stage1_transformer_weights, restore_checkpoint, save_checkpoint
    from aether.training.simgen_dataset import FixedViewSimGenDataset
    from aether.training.stage2_config import load_stage2_training_config
    from aether.training.stage2_losses import compute_stage2_losses, reconstruct_clean_latents
    from scripts.train_fixed_view_simgen import _load_training_components, accelerator_options, diffusion_training_target, optimizer_update, save_fixed_rollout_artifacts, unwrapped_transformer_config

    config = load_stage2_training_config(args.config, args.override)
    if args.preflight:
        preflight(config)
        return
    accelerator = Accelerator(**accelerator_options(config))
    dataset = FixedViewSimGenDataset(config.data_root, config.sample_ids)
    dataloader = DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=config.dataloader_num_workers, pin_memory=config.pin_memory)
    pipeline, transformer, vae, scheduler, prompts = _load_training_components(config, accelerator)
    load_stage1_transformer_weights(transformer, config.stage1_checkpoint)
    optimizer = torch.optim.AdamW(transformer.parameters(), lr=config.learning_rate, betas=(config.adam_beta1, config.adam_beta2), eps=config.adam_epsilon, weight_decay=config.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config.learning_rate, total_steps=config.max_train_steps, pct_start=config.onecycle_pct_start)
    transformer, optimizer, dataloader, lr_scheduler = accelerator.prepare(transformer, optimizer, dataloader, lr_scheduler)
    transformer_config = unwrapped_transformer_config(transformer, accelerator)
    manifest = {"schema_version": 1, "objective": "stage2_image_space_refinement", "stage1_checkpoint": str(config.stage1_checkpoint)}
    checkpoint = None if args.gpu_smoke_test else (latest_checkpoint(config.output_dir) if args.resume in (None, "latest") else Path(args.resume))
    step = restore_checkpoint(accelerator, checkpoint, manifest) if checkpoint else 0
    weights = None
    while step < (1 if args.gpu_smoke_test else config.max_train_steps):
        for batch in dataloader:
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
                losses = compute_stage2_losses(vae, reconstruct_clean_latents(scheduler, noisy, prediction, timesteps), batch)
                weights = weights or _weights(config, mse, losses, accelerator)
                total = mse + weights["rgb"] * losses.rgb + weights["depth"] * losses.depth + weights["pointmap"] * losses.pointmap
                accelerator.backward(total)
                optimizer_update(accelerator, optimizer, lr_scheduler, transformer, config.max_grad_norm)
            if accelerator.sync_gradients:
                step += 1
                accelerator.log({"train/loss": total.item(), "train/mse": mse.item(), "train/rgb_ms_ssim": losses.rgb.item(), "train/depth_ssi": losses.depth.item(), "train/pointmap": losses.pointmap.item()}, step=step)
                if step % config.output_interval == 0 or step == config.max_train_steps:
                    save_checkpoint(accelerator, config.output_dir, step, manifest, keep_last=2)
                    if accelerator.is_main_process:
                        save_fixed_rollout_artifacts(accelerator, config, dataset, pipeline, transformer, vae, scheduler, prompts, step)
                if step >= (1 if args.gpu_smoke_test else config.max_train_steps):
                    return


if __name__ == "__main__":
    main()
