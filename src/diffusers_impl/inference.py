from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .pipeline_crs import CRSDiffusionPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CRS inference with diffusers-style pipeline.")
    parser.add_argument("--config", default="./configs/crs.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative_prompt", default="Low resolution, cropped, worst quality, low quality")
    parser.add_argument("--local_control", required=True, help="Path to .npy local control tensor shaped [H,W,18] or [B,H,W,18]")
    parser.add_argument("--global_control", required=True, help="Path to .npy global control tensor shaped [1536] or [B,1536]")
    parser.add_argument("--metadata", default="0,0,0,0,0,0,0", help="Comma-separated 7 metadata values")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--global_strength", type=float, default=1.0)
    parser.add_argument("--num_images", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="./outputs_diffusers")
    return parser.parse_args()


def _load_local_control(path: str, device: torch.device, dtype: torch.dtype, batch_size: int) -> torch.Tensor:
    arr = np.load(path)
    if arr.ndim == 3:
        arr = arr[None, ...]
    tensor = torch.from_numpy(arr).to(device=device, dtype=dtype)
    tensor = tensor.permute(0, 3, 1, 2).contiguous() / 255.0
    if tensor.shape[0] != batch_size:
        tensor = tensor.expand(batch_size, *tensor.shape[1:]).contiguous()
    return tensor


def _load_global_control(path: str, device: torch.device, dtype: torch.dtype, batch_size: int) -> torch.Tensor:
    arr = np.load(path)
    if arr.ndim == 1:
        arr = arr[None, ...]
    tensor = torch.from_numpy(arr).to(device=device, dtype=dtype)
    if tensor.shape[0] != batch_size:
        tensor = tensor.expand(batch_size, -1).contiguous()
    return tensor


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    pipe = CRSDiffusionPipeline.from_legacy_checkpoint(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=str(device),
        torch_dtype=dtype,
    )

    batch_size = args.num_images
    local_control = _load_local_control(args.local_control, device=device, dtype=dtype, batch_size=batch_size)
    global_control = _load_global_control(args.global_control, device=device, dtype=dtype, batch_size=batch_size)
    metadata = [float(x.strip()) for x in args.metadata.split(",")]

    generator = torch.Generator(device=device).manual_seed(args.seed)
    output = pipe(
        prompt=[args.prompt],
        negative_prompt=[args.negative_prompt],
        local_control=local_control,
        global_control=global_control,
        metadata=metadata,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        global_strength=args.global_strength,
        num_images_per_prompt=args.num_images,
        height=args.height,
        width=args.width,
        generator=generator,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for idx, image in enumerate(output.images):
        image.save(output_dir / f"sample_{idx:03d}.png")


if __name__ == "__main__":
    main()
