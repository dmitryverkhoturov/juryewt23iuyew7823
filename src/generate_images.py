from __future__ import annotations
from pathlib import Path
from typing import List
import torch
from PIL import Image
from utils import load_pipe, parse_args

if __name__ == "__main__":
    args = parse_args()

    def make_gen(seed, device):
        return torch.Generator(device=("cuda" if torch.cuda.is_available() else "cpu")).manual_seed(seed)

    # 1) BASE IMAGE (no LoRA), fixed seed
    seed = 1234
    base_pipe = load_pipe(lora_path=None, device=args.device)
    g = make_gen(seed, base_pipe.device)
    base_img = base_pipe(args.prompt, num_inference_steps=35, generator=g).images[0]
    Path("outputs/base_gen").mkdir(parents=True, exist_ok=True)
    base_img.save("outputs/base_gen/seed1234.png")

    # 2) LoRA IMAGE (with adapter), same seed
    pipe = load_pipe(args.lora_path, args.device)  # your usual LoRA pipe
    g = make_gen(seed, pipe.device)
    lora_img = pipe(args.prompt, num_inference_steps=35, generator=g).images[0]
    Path("outputs/gen").mkdir(parents=True, exist_ok=True)
    lora_img.save("outputs/gen/seed1234_lora.png")

    # 3) (optional) generate the rest normally
    for i in range(args.num_images):
        img = pipe(args.prompt, num_inference_steps=35).images[0]
        img.save(Path(args.output_dir) / f"img_{i}.png")
