from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import List, Optional, Sequence

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXTERNAL_DIFFUSERS_SRC = _REPO_ROOT / "external" / "diffusers" / "src"
if _EXTERNAL_DIFFUSERS_SRC.exists() and str(_EXTERNAL_DIFFUSERS_SRC) not in sys.path:
    sys.path.insert(0, str(_EXTERNAL_DIFFUSERS_SRC))

from diffusers import DDIMScheduler, DiffusionPipeline
from diffusers.pipelines.pipeline_utils import BaseOutput
from PIL import Image

from .modeling_crs import CRSDiffusersWrapper


@dataclass
class CRSPipelineOutput(BaseOutput):
    images: List[Image.Image]
    latents: torch.Tensor


class CRSDiffusionPipeline(DiffusionPipeline):
    model_cpu_offload_seq = "model"

    def __init__(self, model: CRSDiffusersWrapper, scheduler: DDIMScheduler):
        super().__init__()
        self.register_modules(model=model, scheduler=scheduler)

    @classmethod
    def from_legacy_checkpoint(
        cls,
        config_path: str,
        checkpoint_path: str,
        device: str = "cpu",
        torch_dtype: torch.dtype = torch.float32,
    ) -> "CRSDiffusionPipeline":
        model = CRSDiffusersWrapper.from_legacy_checkpoint(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            device=device,
            torch_dtype=torch_dtype,
        )
        scheduler = DDIMScheduler(
            beta_start=float(getattr(model.legacy_model, "linear_start", 0.00085)),
            beta_end=float(getattr(model.legacy_model, "linear_end", 0.0120)),
            beta_schedule="scaled_linear",
            num_train_timesteps=int(getattr(model.legacy_model, "num_timesteps", 1000)),
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
        )
        return cls(model=model, scheduler=scheduler).to(device=device)

    @torch.no_grad()
    def __call__(
        self,
        prompt: str | Sequence[str],
        local_control: torch.Tensor,
        global_control: torch.Tensor,
        metadata: torch.Tensor | List[float],
        negative_prompt: str | Sequence[str] = "",
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        global_strength: float = 1.0,
        height: int = 512,
        width: int = 512,
        num_images_per_prompt: int = 1,
        generator: Optional[torch.Generator] = None,
        eta: float = 0.0,
    ) -> CRSPipelineOutput:
        if isinstance(prompt, str):
            prompt = [prompt]
        batch_size = len(prompt)
        device = self._execution_device

        prompt_embeds = self.model.encode_prompt(prompt, num_images_per_prompt=num_images_per_prompt).to(device)
        negative_prompt = [negative_prompt] * batch_size if isinstance(negative_prompt, str) else list(negative_prompt)
        negative_embeds = self.model.encode_prompt(negative_prompt, num_images_per_prompt=num_images_per_prompt).to(device)

        total_batch = batch_size * num_images_per_prompt
        local_control = local_control.to(device=device, dtype=prompt_embeds.dtype)
        global_control = global_control.to(device=device, dtype=prompt_embeds.dtype)
        if local_control.shape[0] != total_batch:
            local_control = local_control.expand(total_batch, *local_control.shape[1:]).contiguous()
        if global_control.shape[0] != total_batch:
            global_control = global_control.expand(total_batch, -1).contiguous()

        metadata = self.model.prepare_metadata(metadata=metadata, batch_size=total_batch, device=device)

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        latents = torch.randn(
            (total_batch, 4, height // 8, width // 8),
            generator=generator,
            device=device,
            dtype=prompt_embeds.dtype,
        )
        latents *= self.scheduler.init_noise_sigma

        for t in self.scheduler.timesteps:
            latent_model_input = self.scheduler.scale_model_input(latents, t)
            noise_pred_text = self.model(
                latents=latent_model_input,
                timesteps=t,
                prompt_embeds=prompt_embeds,
                local_control=local_control,
                global_control=global_control,
                metadata=metadata,
                global_strength=global_strength,
            )

            if guidance_scale > 1.0:
                uc_local_control, uc_global_control = self.model.zeros_like_conditions(local_control, global_control)
                noise_pred_uncond = self.model(
                    latents=latent_model_input,
                    timesteps=t,
                    prompt_embeds=negative_embeds,
                    local_control=uc_local_control,
                    global_control=uc_global_control,
                    metadata=metadata,
                    global_strength=global_strength,
                )
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            else:
                noise_pred = noise_pred_text

            latents = self.scheduler.step(noise_pred, t, latents, eta=eta, return_dict=True).prev_sample

        decoded = self.model.decode_latents(latents)
        image_tensor = (decoded / 2 + 0.5).clamp(0, 1)
        image_array = image_tensor.detach().cpu().permute(0, 2, 3, 1).float().numpy()
        pil_images = [Image.fromarray((img * 255).round().astype(np.uint8)) for img in image_array]

        return CRSPipelineOutput(images=pil_images, latents=latents)
