from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from accelerate import Accelerator

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXTERNAL_DIFFUSERS_SRC = _REPO_ROOT / "external" / "diffusers" / "src"
if _EXTERNAL_DIFFUSERS_SRC.exists() and str(_EXTERNAL_DIFFUSERS_SRC) not in sys.path:
    sys.path.insert(0, str(_EXTERNAL_DIFFUSERS_SRC))

from diffusers import DDPMScheduler
from torch.utils.data import DataLoader, Dataset

from .modeling_crs import CRSDiffusersWrapper


class CRSTrainDataset(Dataset):
    def __init__(self, path: str):
        self.items: List[Dict[str, Any]] = torch.load(path, map_location="cpu")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.items[index]


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    images = torch.stack([torch.as_tensor(x["image"]).float() for x in batch], dim=0)
    if images.ndim == 4 and images.shape[-1] in (1, 3):
        images = images.permute(0, 3, 1, 2).contiguous()
    images = images / 127.5 - 1.0

    local = torch.stack([torch.as_tensor(x["local_control"]).float() for x in batch], dim=0)
    if local.ndim == 4 and local.shape[-1] > 1:
        local = local.permute(0, 3, 1, 2).contiguous()
    local = local / 255.0

    global_control = torch.stack([torch.as_tensor(x["global_control"]).float() for x in batch], dim=0)
    metadata = torch.stack([torch.as_tensor(x["metadata"]).float() for x in batch], dim=0)
    prompts = [str(x["prompt"]) for x in batch]

    return {
        "images": images,
        "local_control": local,
        "global_control": global_control,
        "metadata": metadata,
        "prompts": prompts,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CRS model with diffusers/accelerate loop.")
    parser.add_argument("--config", default="./configs/crs.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train_data", required=True, help="Path to torch-serialized list[dict] training set")
    parser.add_argument("--output_dir", default="./outputs_diffusers_train")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--save_every_steps", type=int, default=500)
    parser.add_argument("--mixed_precision", default="no", choices=["no", "fp16", "bf16"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    accelerator = Accelerator(mixed_precision=args.mixed_precision)

    model = CRSDiffusersWrapper.from_legacy_checkpoint(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device="cpu",
        torch_dtype=torch.float32,
    )

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=int(getattr(model.legacy_model, "num_timesteps", 1000)),
        beta_start=float(getattr(model.legacy_model, "linear_start", 0.00085)),
        beta_end=float(getattr(model.legacy_model, "linear_end", 0.0120)),
        beta_schedule="scaled_linear",
        prediction_type="epsilon",
    )

    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=args.learning_rate)
    dataset = CRSTrainDataset(args.train_data)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    model.train()

    global_step = 0
    for epoch in range(args.num_epochs):
        for batch in dataloader:
            with accelerator.accumulate(model):
                latents = model.encode_images(batch["images"]).detach()
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (latents.shape[0],),
                    device=latents.device,
                    dtype=torch.long,
                )
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                prompt_embeds = model.encode_prompt(batch["prompts"]).to(latents.device)
                metadata = batch["metadata"].to(latents.device)
                local_control = batch["local_control"].to(latents.device)
                global_control = batch["global_control"].to(latents.device)

                noise_pred = model(
                    latents=noisy_latents,
                    timesteps=timesteps,
                    prompt_embeds=prompt_embeds,
                    local_control=local_control,
                    global_control=global_control,
                    metadata=metadata,
                )
                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.is_main_process and global_step > 0 and global_step % args.save_every_steps == 0:
                save_dir = Path(args.output_dir) / f"checkpoint-epoch{epoch}-step{global_step}"
                accelerator.unwrap_model(model).save_pretrained(str(save_dir), safe_serialization=True)
            global_step += 1

    if accelerator.is_main_process:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        accelerator.unwrap_model(model).save_pretrained(str(Path(args.output_dir) / "final"), safe_serialization=True)


if __name__ == "__main__":
    main()
