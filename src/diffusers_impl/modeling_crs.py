from __future__ import annotations

from pathlib import Path
from typing import List, Sequence
import sys

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXTERNAL_DIFFUSERS_SRC = _REPO_ROOT / "external" / "diffusers" / "src"
if _EXTERNAL_DIFFUSERS_SRC.exists() and str(_EXTERNAL_DIFFUSERS_SRC) not in sys.path:
    sys.path.insert(0, str(_EXTERNAL_DIFFUSERS_SRC))

from diffusers import ConfigMixin, ModelMixin
from diffusers.configuration_utils import register_to_config

from models.util import create_model, load_state_dict


class CRSDiffusersWrapper(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self, config_path: str = "./configs/crs.yaml", checkpoint_path: str = ""):
        super().__init__()
        self.legacy_model = create_model(config_path)
        if checkpoint_path:
            self.legacy_model.load_state_dict(load_state_dict(checkpoint_path, location="cpu"), strict=True)

    @classmethod
    def from_legacy_checkpoint(
        cls,
        config_path: str,
        checkpoint_path: str,
        device: str = "cpu",
        torch_dtype: torch.dtype = torch.float32,
    ) -> "CRSDiffusersWrapper":
        model = cls(config_path=config_path, checkpoint_path=checkpoint_path)
        model.to(device=device, dtype=torch_dtype)
        return model

    def trainable_parameters(self):
        return [p for p in self.legacy_model.parameters() if p.requires_grad]

    @torch.no_grad()
    def encode_prompt(self, prompt: Sequence[str], num_images_per_prompt: int = 1) -> torch.Tensor:
        if isinstance(prompt, str):
            prompt = [prompt]
        embeds = self.legacy_model.get_learned_conditioning(list(prompt))
        if num_images_per_prompt > 1:
            embeds = embeds.repeat_interleave(num_images_per_prompt, dim=0)
        return embeds

    @torch.no_grad()
    def encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        posterior = self.legacy_model.encode_first_stage(pixel_values)
        return self.legacy_model.get_first_stage_encoding(posterior)

    @torch.no_grad()
    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        return self.legacy_model.decode_first_stage(latents)

    def forward(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        prompt_embeds: torch.Tensor,
        local_control: torch.Tensor,
        global_control: torch.Tensor,
        metadata: torch.Tensor,
        global_strength: float = 1.0,
    ) -> torch.Tensor:
        if timesteps.ndim == 0:
            timesteps = timesteps[None]
        if timesteps.ndim == 1 and timesteps.shape[0] == 1 and latents.shape[0] > 1:
            timesteps = timesteps.expand(latents.shape[0])

        cond = {
            "c_crossattn": [prompt_embeds],
            "local_control": [local_control],
            "global_control": [global_control],
            "metadata": [metadata],
        }
        return self.legacy_model.apply_model(
            x_noisy=latents,
            t=timesteps,
            cond=cond,
            metadata=metadata,
            global_strength=global_strength,
        )

    @staticmethod
    def zeros_like_conditions(
        local_control: torch.Tensor,
        global_control: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return local_control, torch.zeros_like(global_control)

    @staticmethod
    def prepare_metadata(metadata: torch.Tensor | List[float], batch_size: int, device: torch.device) -> torch.Tensor:
        if not torch.is_tensor(metadata):
            metadata = torch.tensor(metadata, dtype=torch.float32)
        metadata = metadata.to(device=device, dtype=torch.float32)
        if metadata.ndim == 1:
            metadata = metadata.unsqueeze(0)
        if metadata.shape[0] != batch_size:
            metadata = metadata.expand(batch_size, -1)
        return metadata.contiguous()
