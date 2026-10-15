#!/usr/bin/env python3
"""Text-to-image generation benchmark (diffusers): SD 1.5, SD 3.5, FLUX.1/.2, Z-Image.

One iteration = one full pipeline call (text encode + all denoising steps + VAE
decode) for --batch_size images at each model's reference resolution/steps.
The requested precision is what runs; fp16 on MMDiT/FLUX models can overflow,
which the image check below turns into a FAIL instead of a fast black image.
"""

import sys
from pathlib import Path
ROOT = next(p for p in Path(__file__).resolve().parents if (p / "benchmark.py").exists())
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from core.args import build_base_parser
from core.harness import InferenceHarness, run_main
from core.schema import MetricEntry

PROMPT = "A photo of a red fox sitting in a snowy forest at sunrise, highly detailed"

# name: (repo, resolution, steps, guidance, extra from_pretrained kwargs) — settings from each model card
MODELS = {
    "stable_diffusion_1_5": ("stable-diffusion-v1-5/stable-diffusion-v1-5", 512, 20, 7.5,
                             {"safety_checker": None, "requires_safety_checker": False}),
    "stable_diffusion_3_5_medium": ("stabilityai/stable-diffusion-3.5-medium", 1024, 40, 4.5, {}),
    "stable_diffusion_3_5_large_turbo": ("stabilityai/stable-diffusion-3.5-large-turbo", 1024, 4, 0.0, {}),
    "flux_1_schnell": ("black-forest-labs/FLUX.1-schnell", 1024, 4, 0.0, {}),
    "flux_1_dev": ("black-forest-labs/FLUX.1-dev", 1024, 28, 3.5, {}),
    "flux_2_klein_4b": ("black-forest-labs/FLUX.2-klein-4B", 1024, 4, 1.0, {}),
    "z_image_turbo": ("Tongyi-MAI/Z-Image-Turbo", 1024, 9, 0.0, {}),
}


class DiffusionGeneration(InferenceHarness):
    use_case = "generation"
    primary_metric = "seconds_per_image"
    throughput_unit = "images/sec"

    def load_model(self):
        from diffusers import DiffusionPipeline

        repo, res, steps, guidance, extra = MODELS[self.args.model]
        self._res = self.args.resolution or res
        self._steps = self.args.steps or steps
        self._guidance = guidance
        pipe = DiffusionPipeline.from_pretrained(repo, torch_dtype=self.dtype, **extra)
        if self.args.cpu_offload:
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(self.device)
        pipe.set_progress_bar_config(disable=True)
        print(f"{type(pipe).__name__}: {self._res}x{self._res}, {self._steps} steps, guidance {guidance}")
        return pipe

    def prepare_inputs(self):
        return [PROMPT] * self.batch_size

    def run_step(self, pipe, prompts):
        return pipe(prompt=prompts, height=self._res, width=self._res, num_inference_steps=self._steps,
                    guidance_scale=self._guidance, output_type="np",
                    generator=torch.Generator("cpu").manual_seed(0)).images

    def get_extra_metrics(self, pipe, prompts, images):
        self._images = np.asarray(images, dtype=np.float32)
        out = ROOT / "benchmark_results" / "images" / f"{self.args.model}_{self.precision}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        from PIL import Image
        Image.fromarray((np.nan_to_num(self._images[0]).clip(0, 1) * 255).astype(np.uint8)).save(out)
        print(f"Saved sample image: {out}")
        return []

    def finish(self, metrics, stats, validator=None):
        if stats.get("mean"):
            metrics.insert(0, MetricEntry("seconds_per_image", stats["mean"] / 1000 / self.batch_size, "s",
                                          "lower_is_better"))
        return super().finish(metrics, stats, validator)

    def validate_result(self, pipe, prompts, images, validator):
        img = self._images
        validator.expect_equals("images_finite", bool(np.isfinite(img).all()), True)
        validator.expect_greater_than("image_std", float(np.nan_to_num(img).std()), 0.02)  # not blank/black

    def get_config(self):
        return {"resolution": f"{self._res}x{self._res}", "steps": self._steps, "guidance": self._guidance,
                "cpu_offload": self.args.cpu_offload, "pipeline": type(self._model).__name__}


if __name__ == "__main__":
    parser = build_base_parser("Text-to-image generation (diffusers)")
    parser.add_argument("--resolution", type=int, help="square image size (default: model's reference)")
    parser.add_argument("--steps", type=int, help="denoising steps (default: model's reference)")
    parser.add_argument("--cpu_offload", "--cpu-offload", action="store_true",
                        help="model CPU offload for models larger than VRAM (measures PCIe too)")
    parser.set_defaults(model="stable_diffusion_1_5", precision="bf16", num_warmup=1, num_runs=3)
    run_main(DiffusionGeneration, parser)
