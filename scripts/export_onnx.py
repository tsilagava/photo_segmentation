"""Export the trained segmenters to ONNX for the in-browser demo (web/).

    pip install onnx onnxruntime
    python scripts/export_onnx.py                       # weights/v3.pt, weights/v5.pt -> web/models/v3.onnx, v5.onnx

The exported graph takes an RGB image scaled to 0..1, shape (1, 3, H, W) with H and W multiples of 32, does the
ImageNet normalisation itself and returns the pet probability (1, H, W) = 1 - softmax(background), so the browser
only has to resize, run and average. Weights are stored as float16 (half the download) and cast back on load, so
inputs, outputs and the computation stay float32.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
from onnx import helper, numpy_helper

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from petseg.model import load_segmenter  # noqa: E402


class PetProbability(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        return 1 - self.net((x - self.mean) / self.std).softmax(1)[:, 0]


def fp16_storage(m):
    """Store float32 weights as float16 followed by a Cast back to float32: half the file size, while every op
    still computes in float32 (the browser's WASM backend has no float16 kernels for most ops)."""
    keep, casts = [], []
    for init in m.graph.initializer:
        if init.data_type == onnx.TensorProto.FLOAT and np.prod(init.dims) > 16:
            half = numpy_helper.from_array(numpy_helper.to_array(init).astype(np.float16), init.name + "_fp16")
            keep.append(half)
            casts.append(helper.make_node("Cast", [half.name], [init.name], to=onnx.TensorProto.FLOAT))
        else:
            keep.append(init)
    del m.graph.initializer[:]
    m.graph.initializer.extend(keep)
    nodes = casts + list(m.graph.node)
    del m.graph.node[:]
    m.graph.node.extend(nodes)
    onnx.checker.check_model(m)
    return m


def export(src, dst):
    net, size, meta = load_segmenter(src)
    model = PetProbability(net).eval()
    x = torch.rand(1, 3, size, size)
    tmp = Path(dst).with_suffix(".fp32.onnx")
    torch.onnx.export(model, x, tmp, input_names=["image"], output_names=["pet_prob"], opset_version=17,
                      dynamic_axes={"image": {2: "height", 3: "width"}, "pet_prob": {1: "height", 2: "width"}},
                      dynamo=False)
    onnx.save(fp16_storage(onnx.load(tmp)), dst)
    tmp.unlink()

    # the ONNX model must agree with PyTorch, also at another input size
    sess = ort.InferenceSession(str(dst), providers=["CPUExecutionProvider"])
    for s in (size, size + 64):
        xi = torch.rand(1, 3, s, s)
        with torch.no_grad():
            ref = model(xi).numpy()
        got = sess.run(None, {"image": xi.numpy()})[0]
        print(f"  {s}px: max |onnx - torch| = {np.abs(got - ref).max():.4f}, "
              f"mask agreement {((got > 0.5) == (ref > 0.5)).mean():.4%}")
    print(f"{dst}: input {size} px, {Path(dst).stat().st_size / 1e6:.1f} MB")
    return size


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", nargs="*", default=["weights/v3.pt", "weights/v5.pt"])
    p.add_argument("--out", default="web/models")
    a = p.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    for w in a.weights:
        export(w, Path(a.out) / f"{Path(w).stem}.onnx")


if __name__ == "__main__":
    main()
