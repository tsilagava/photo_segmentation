"""Segment new images from the command line.

    python predict.py photo.jpg more_photos/ --out predictions/
    python predict.py photo.jpg --matting          # soft edges with ViTMatte (downloads the model on first use)

For every image writes <name>_mask.png (white = pet), <name>_overlay.jpg and <name>_cutout.png (transparent background).
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from petseg.predict import PetSegmenter, cutout, find_weights, overlay

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inputs", nargs="+", help="image files or folders")
    p.add_argument("--out", default="predictions")
    p.add_argument("--weights", nargs="*", help="checkpoint files (default: every weights/*.pt, ensembled)")
    p.add_argument("--no-tta", action="store_true")
    p.add_argument("--matting", action="store_true", help="refine the edge with ViTMatte")
    p.add_argument("--device", default=None)
    a = p.parse_args()

    files = []
    for i in map(Path, a.inputs):
        files += sorted(f for f in i.iterdir() if f.suffix.lower() in EXTS) if i.is_dir() else [i]
    seg = PetSegmenter({Path(w).stem: w for w in a.weights} if a.weights else find_weights(), device=a.device)
    print("models:", ", ".join(seg.models), "| device:", seg.device)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for f in files:
        r = seg.predict(f, tta=not a.no_tta, matting=a.matting)
        Image.fromarray((r["mask"] * 255).astype(np.uint8)).save(out / f"{f.stem}_mask.png")
        Image.fromarray(overlay(r["image"], r["alpha"])).save(out / f"{f.stem}_overlay.jpg", quality=92)
        Image.fromarray(cutout(r["image"], r["alpha"])).save(out / f"{f.stem}_cutout.png")
        print(f"{f.name}: pet covers {r['pet_share']:.1%} of the image")
    print("saved to", out)


if __name__ == "__main__":
    main()
