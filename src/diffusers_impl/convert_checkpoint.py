from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .pipeline_crs import CRSDiffusionPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert CRS legacy checkpoint into diffusers save_pretrained format.")
    parser.add_argument("--config", required=True, help="Path to legacy YAML config (e.g. ./configs/crs.yaml)")
    parser.add_argument("--checkpoint", required=True, help="Path to legacy checkpoint (.ckpt/.safetensors)")
    parser.add_argument("--output_dir", required=True, help="Output directory for diffusers-format artifacts")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]

    pipe = CRSDiffusionPipeline.from_legacy_checkpoint(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device="cpu",
        torch_dtype=dtype,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pipe.save_pretrained(str(output_dir), safe_serialization=True)

    conversion_meta = {
        "source_config": str(Path(args.config).resolve()),
        "source_checkpoint": str(Path(args.checkpoint).resolve()),
        "pipeline_class": "CRSDiffusionPipeline",
        "notes": "This conversion keeps CRS architecture and exports it in diffusers save_pretrained layout.",
    }
    (output_dir / "conversion_metadata.json").write_text(json.dumps(conversion_meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
