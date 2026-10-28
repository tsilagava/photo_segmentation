"""Turn a training checkpoint (best.pt from train.py or the Colab notebooks) into a small inference file.

    python scripts/export_weights.py "/path/to/checkpoints_v5/best.pt" weights/v5.pt
    python scripts/export_weights.py "/path/to/checkpoints_v3/best.pt" weights/v3.pt

Keeps only the model weights (stored as float16, about 50 MB) plus config, class names and epoch;
loading converts back to float32.
"""
import argparse
import hashlib
from pathlib import Path

import torch


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint")
    p.add_argument("out")
    p.add_argument("--fp32", action="store_true", help="keep float32 weights (about 100 MB)")
    a = p.parse_args()

    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    sd = {k: (v.half() if v.is_floating_point() and not a.fp32 else v) for k, v in ck["model"].items()}
    out = {"model": sd, "config": dict(ck["config"]), "class_names": list(ck["class_names"]), "epoch": int(ck["epoch"])}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, a.out)
    digest = hashlib.sha256(Path(a.out).read_bytes()).hexdigest()
    print(f"{a.out}: epoch {out['epoch']}, input {out['config']['img_size']} px, "
          f"{Path(a.out).stat().st_size / 1e6:.1f} MB, sha256 {digest}")


if __name__ == "__main__":
    main()
