"""Inference: v3 + v5 ensemble with test-time augmentation, automatic trimap and optional ViTMatte soft edges.

Same pipeline as notebooks/Pet_Edges_Ensemble_ViTMatte.ipynb:
1. every segmenter runs at its own training size with TTA (horizontal flip x scales 0.75 / 1 / 1.25), the pet
   probabilities are resized to the image and averaged, first over TTA, then over the models;
2. the 0.5-thresholded mask is shrunk (sure pet) and grown (sure background), the ring in between is "unknown";
3. ViTMatte turns the unknown ring into a soft alpha matte (optional, needs `transformers`).
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage

from .model import load_segmenter

MATTE_MODEL = "hustvl/vitmatte-small-composition-1k"
TTA_SCALES = [0.75, 1.0, 1.25]


def default_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def find_weights(weights_dir="weights"):
    """{model name: path} for every *.pt file in weights_dir (e.g. weights/v3.pt, weights/v5.pt)."""
    return {p.stem: p for p in sorted(Path(weights_dir).glob("*.pt"))}


class PetSegmenter:
    def __init__(self, weights, device=None, ensemble_weights=None, band_frac=0.02, band_min_px=4,
                 max_side=1024, matte_model=MATTE_MODEL):
        """weights: {name: checkpoint path} (or a list of paths). ensemble_weights default to equal weights."""
        if not isinstance(weights, dict):
            weights = {Path(p).stem: p for p in weights}
        if not weights:
            raise FileNotFoundError("no model weights given: put v3.pt / v5.pt into weights/ (see weights/README.md)")
        self.device = torch.device(device) if device else default_device()
        self.models = {}
        for name, path in weights.items():
            net, size, meta = load_segmenter(path, self.device)
            self.models[name] = (net, size)
        ew = ensemble_weights or {n: 1 / len(self.models) for n in self.models}
        total = sum(ew[n] for n in self.models)
        self.ensemble_weights = {n: ew[n] / total for n in self.models}
        self.band_frac, self.band_min_px, self.max_side = band_frac, band_min_px, max_side
        self.matte_model = matte_model
        self._matte = None
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

    # ---- segmentation ----
    @torch.no_grad()
    def _model_prob(self, net, native, x, h, w, tta):
        acc, n = 0, 0
        amp = self.device.type == "cuda"
        for s in (TTA_SCALES if tta else [1.0]):
            sz = max(64, int(round(native * s / 32)) * 32)
            xi = (F.interpolate(x, size=(sz, sz), mode="bilinear", antialias=True, align_corners=False) - self.mean) / self.std
            for flip in ((False, True) if tta else (False,)):
                with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=amp):
                    lo = net(xi.flip(-1) if flip else xi).float()
                lo = lo.flip(-1) if flip else lo
                # pet = everything but background, so 3-class (boundary) checkpoints work too
                acc = acc + 1 - F.interpolate(lo, size=(h, w), mode="bilinear", align_corners=False).softmax(1)[0, 0]
                n += 1
        return (acc / n).cpu().numpy()

    def model_probs(self, img, tta=True):
        """{model name: HxW pet probability} for an HxWx3 uint8 RGB image."""
        h, w = img.shape[:2]
        x = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)[None].float().div(255).to(self.device)
        return {name: self._model_prob(net, native, x, h, w, tta) for name, (net, native) in self.models.items()}

    def pet_probability(self, img, tta=True):
        return sum(p * self.ensemble_weights[m] for m, p in self.model_probs(img, tta).items())

    # ---- soft edges ----
    def band_px(self, shape):
        return max(self.band_min_px, int(round(self.band_frac * max(shape))))

    def auto_trimap(self, mask):
        r = self.band_px(mask.shape)
        disk = np.hypot(*np.mgrid[-r:r + 1, -r:r + 1]) <= r
        tri = np.full(mask.shape, 128, np.uint8)
        tri[ndimage.binary_erosion(mask, disk)] = 255
        tri[~ndimage.binary_dilation(mask, disk)] = 0
        return tri

    def _load_matte(self):
        if self._matte is None:
            from transformers import VitMatteForImageMatting, VitMatteImageProcessor
            self._matte = (VitMatteImageProcessor.from_pretrained(self.matte_model),
                           VitMatteForImageMatting.from_pretrained(self.matte_model).to(self.device).eval())
        return self._matte

    @torch.no_grad()
    def alpha_matte(self, img, tri):
        processor, matte = self._load_matte()
        inputs = processor(images=Image.fromarray(img), trimaps=Image.fromarray(tri), return_tensors="pt").to(self.device)
        a = matte(**inputs).alphas[0, 0, :img.shape[0], :img.shape[1]].float().cpu().numpy()
        a[tri == 255], a[tri == 0] = 1.0, 0.0      # keep the sure regions exactly as the segmenter decided
        return a.clip(0, 1)

    # ---- everything ----
    def prepare(self, image):
        """PIL image / path / array -> HxWx3 uint8 RGB, long side capped at max_side."""
        if isinstance(image, (str, Path)):
            image = Image.open(image)
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        image = image.convert("RGB")
        if self.max_side and max(image.size) > self.max_side:
            image.thumbnail((self.max_side, self.max_side), Image.LANCZOS)
        return np.array(image)

    def predict(self, image, tta=True, matting=False):
        """Returns a dict with image, prob (HxW float), mask (HxW bool), alpha (HxW float), trimap and pet_share."""
        img = self.prepare(image)
        prob = self.pet_probability(img, tta)
        mask = prob > 0.5
        out = {"image": img, "prob": prob, "mask": mask, "alpha": mask.astype(np.float32), "trimap": None}
        if matting and mask.any():
            out["trimap"] = self.auto_trimap(mask)
            out["alpha"] = self.alpha_matte(img, out["trimap"])
        out["pet_share"] = float(mask.mean())
        return out


def overlay(img, mask, color=(228, 26, 28), alpha=0.45):
    """Pet pixels tinted red, with the mask outline drawn on top."""
    out = img.astype(np.float32).copy()
    m = np.asarray(mask) > 0.5
    out[m] = (1 - alpha) * out[m] + alpha * np.array(color)
    edge = m & ~ndimage.binary_erosion(m, iterations=max(1, round(max(m.shape) / 400)))
    out[edge] = (255, 255, 255)
    return out.astype(np.uint8)


def cutout(img, alpha):
    """RGBA image: the pet on a transparent background."""
    return np.dstack([img, (np.asarray(alpha) * 255).round().astype(np.uint8)])
