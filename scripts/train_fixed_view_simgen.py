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
        help="Load configured weights and data, run one real CUDA optimizer step, then exit.",
    )
    parser.add_argument(
        "--resume",
        default="latest",
        help="Checkpoint directory to restore, or 'latest' to resume the newest output checkpoint.",
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


def training_steps(max_train_steps: int, gpu_smoke_test: bool) -> int:
    """Keep the saved run configuration intact while capping GPU smoke mode."""
    return 1 if gpu_smoke_test else max_train_steps


def has_remaining_steps(completed_steps: int, max_train_steps: int) -> bool:
    return completed_steps < max_train_steps


def run_training(config, gpu_smoke_test: bool = False, resume: str | None = "latest") -> None:
    import torch
    import torch.nn.functional as functional
    from accelerate import Accelerator
    from torch.utils.data import DataLoader

    from aether.training.aether_latents import assemble_aether_training_batch
    from aether.training.checkpointing import (
        latest_checkpoint,
        restore_checkpoint,
        save_checkpoint,
    )

    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
    )
    if gpu_smoke_test and (not torch.cuda.is_available() or accelerator.device.type != "cuda"):
        raise RuntimeError("--gpu-smoke-test requires an available CUDA GPU")
    torch.manual_seed(config.seed)
    max_train_steps = training_steps(config.max_train_steps, gpu_smoke_test)
    dataset = SimGenFixedViewDataset(config.data_root, config.sample_ids)
    dataloader = DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)
    pipeline, transformer, vae, scheduler, prompt_embeds = _load_training_components(
        config, accelerator
    )
    if hasattr(transformer, "enable_gradient_checkpointing"):
        transformer.enable_gradient_checkpointing()
    optimizer = torch.optim.AdamW(transformer.parameters(), lr=config.learning_rate)
    transformer, optimizer, dataloader = accelerator.prepare(transformer, optimizer, dataloader)
    transformer.train()
    if resume == "latest":
        checkpoint = latest_checkpoint(config.output_dir)
    elif resume:
        checkpoint = Path(resume)
    else:
        checkpoint = None
    completed_steps = 0
    if checkpoint is not None:
        completed_steps = restore_checkpoint(accelerator, checkpoint)
        accelerator.print(f"resumed checkpoint {checkpoint} at step {completed_steps}")
    if not has_remaining_steps(completed_steps, max_train_steps):
        accelerator.print(f"training already completed at step {completed_steps}")
        return

    for global_step, batch in enumerate(
        _infinite_batches(dataloader), start=completed_steps + 1
    ):
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
                if transformer.config.use_rotary_positional_embeddings
                else None
            )
            ofs = (
                None
                if transformer.config.ofs_embed_dim is None
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
            loss = functional.mse_loss(prediction.float(), noise.float())
            if not torch.isfinite(loss):
                raise RuntimeError("Stage-1 loss became non-finite")
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if accelerator.sync_gradients and accelerator.is_main_process:
            accelerator.print(f"step={global_step} stage1_mse={loss.item():.6f}")
        if accelerator.sync_gradients and global_step % config.output_interval == 0:
            checkpoint = save_checkpoint(accelerator, config.output_dir, global_step)
            if accelerator.is_main_process:
                accelerator.print(f"saved checkpoint: {checkpoint}")
        if global_step >= max_train_steps:
            break
    if gpu_smoke_test:
        accelerator.print("GPU smoke test passed: one finite Stage-1 optimizer step completed")


def main() -> None:
    args = parse_args()
    config = load_training_config(args.config, args.override)
    if args.dry_run:
        dataset = SimGenFixedViewDataset(config.data_root, config.sample_ids)
        print(f"validated {len(dataset)} fixed-view SimGen samples")
        return
    if args.smoke_test:
        _run_smoke_test()
        return
    run_training(config, gpu_smoke_test=args.gpu_smoke_test, resume=args.resume)


if __name__ == "__main__":
    main()
