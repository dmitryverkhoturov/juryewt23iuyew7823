# generate_base.py
from __future__ import annotations
from pathlib import Path
import argparse
import torch
from diffusers import StableDiffusionPipeline

def main():
    p = argparse.ArgumentParser("Generate images with the base SD1.5 model (no LoRA)")
    p.add_argument("--prompt", required=True, help="Text prompt")
    p.add_argument("--model", default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--num-images", type=int, default=4)
    p.add_argument("--steps", type=int, default=35)
    p.add_argument("--guidance", type=float, default=7.5)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default=None, help="cuda | cpu (auto if not set)")
    p.add_argument("--output-dir", default="outputs/base_gen")
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(args.model, torch_dtype=dtype)
    pipe = pipe.to(device)
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass

    g = None
    if args.seed is not None:
        g = torch.Generator(device=device).manual_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(args.num_images):
        result = pipe(
            args.prompt,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            height=args.height,
            width=args.width,
            generator=g,
        )
        img = result.images[0]
        img.save(out_dir / f"img_{i}.png")

    print(f"Saved {args.num_images} image(s) to {out_dir.resolve()}")

if __name__ == "__main__":
    main()
