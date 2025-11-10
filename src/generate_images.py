from __future__ import annotations
from pathlib import Path
import os
import torch
from PIL import Image
from utils import load_pipe, parse_args

def make_seed_list(n: int, master_seed: int | None) -> list[int]:
    g = torch.Generator(device="cpu")
    g.manual_seed(master_seed if master_seed is not None else int.from_bytes(os.urandom(8), "big"))
    return [int(torch.randint(0, 2**31 - 1, (1,), generator=g).item()) for _ in range(n)]

if __name__ == "__main__":
    args = parse_args()

    out_dir  = Path(args.output_dir)
    out_base = out_dir / "base"
    out_lora = out_dir / "lora"
    out_base.mkdir(parents=True, exist_ok=True)
    out_lora.mkdir(parents=True, exist_ok=True)

    seeds = make_seed_list(args.num_images, args.seed)

    # BASE first (only if --compare_base)
    if args.compare_base:
        base_pipe = load_pipe(lora_path=None, device=args.device)
        for s in seeds:
            g = torch.Generator(device=base_pipe.device).manual_seed(s)
            img: Image.Image = base_pipe(args.prompt, num_inference_steps=35, generator=g).images[0]
            img.save(out_base / f"{s}.png")

    # Then LoRA with the same seeds
    pipe = load_pipe(args.lora_path, args.device)
    for s in seeds:
        g = torch.Generator(device=pipe.device).manual_seed(s)
        img: Image.Image = pipe(args.prompt, num_inference_steps=35, generator=g).images[0]
        img.save(out_lora / f"{s}.png")
