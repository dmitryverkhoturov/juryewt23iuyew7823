from typing import List, Optional

import torch
from diffusers import StableDiffusionPipeline
from peft import PeftModel, PeftConfig
from PIL import Image

import argparse

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _safe_enable_xformers(pipe: StableDiffusionPipeline) -> None:
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass


def load_pipe(
        lora_path,
        device,
        base_model = "runwayml/stable-diffusion-v1-5"):
    logger.info("start load pipeline")
    base_model = "runwayml/stable-diffusion-v1-5"
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(base_model, torch_dtype=dtype)
    pipe = pipe.to(device)
    _safe_enable_xformers(pipe)

    if lora_path:
        logger.info(f"Loading LoRA from: {lora_path}")
        config = PeftConfig.from_pretrained(lora_path)
        pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_path, config=config)
        pipe.unet.eval()
        # ✅ ensure adapter is active and not silently ignored
        try:
            active = getattr(pipe.unet, "active_adapter", None) or "default"
            pipe.unet.set_adapter(active, lora_scale=1.5)  # bump to 1.2–1.5 if still too weak
            # quick sanity log: check a known LoRA key exists
            has_lora = any("lora_" in k for k in dict(pipe.unet.named_parameters()))
            logger.info(f"LoRA active='{active}', scale=1.0, params_with_lora={has_lora}")
        except Exception as e:
            logger.warning(f"Could not set adapter scale (non-fatal): {e}")

    logger.warning("load pipeline done")
    return pipe


def parse_args():
    p = argparse.ArgumentParser(description="Generate images with (optionally) LoRA adapter")
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--lora_path", type=str, help="Path to folder with LoRA weights (peft model)")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--num_images", type=int, default=4)
    p.add_argument("--output_dir", type=str, default="outputs/gen")
    p.add_argument("--compare_base", action="store_true", help="Also generate base images with the same seeds for A/B comparison.")
    p.add_argument("--seed", type=int, default=None, help="Master RNG seed; if omitted, seeds are random.")
    return p.parse_args()