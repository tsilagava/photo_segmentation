"""Gradio demo: upload a photo of a cat or dog, get the mask, an overlay and a transparent cut-out.

    python app.py            # http://127.0.0.1:7860   (add --share for a temporary public link)

Also the entry point for a Hugging Face Space (SDK: gradio). api.py serves the same UI at /ui next to the REST API.
"""
import argparse
import os

import gradio as gr
import numpy as np

from petseg.predict import PetSegmenter, cutout, find_weights, overlay

_segmenter = None


def get_segmenter():
    global _segmenter
    if _segmenter is None:
        _segmenter = PetSegmenter(find_weights(os.environ.get("WEIGHTS_DIR", "weights")))
    return _segmenter


def run(image, tta, matting):
    if image is None:
        raise gr.Error("Upload an image first.")
    seg = get_segmenter()
    r = seg.predict(image, tta=tta, matting=matting)
    mask = (r["alpha"] * 255).round().astype(np.uint8)
    info = (f"**Pet covers {r['pet_share']:.1%} of the image** · {r['image'].shape[1]}×{r['image'].shape[0]} px · "
            f"models: {' + '.join(seg.models)}{' (TTA)' if tta else ''}{' · ViTMatte edges' if matting else ''} · "
            f"device: {seg.device}")
    if not r["mask"].any():
        info += "\n\nNo pet found in this image."
    return overlay(r["image"], r["alpha"]), mask, cutout(r["image"], r["alpha"]), info


def build_demo():
    with gr.Blocks(title="Pet segmentation") as demo:
        gr.Markdown("# Pet segmentation\nU-Net / ResNet-34 trained on Oxford-IIIT Pet (v3 + v5 ensemble). "
                    "Upload a photo of a cat or a dog.")
        with gr.Row():
            with gr.Column():
                inp = gr.Image(type="pil", label="Photo")
                tta = gr.Checkbox(value=True, label="Test-time augmentation (flip × 3 scales)")
                matting = gr.Checkbox(value=False, label="Soft edges with ViTMatte (slower, downloads ~100 MB once)")
                btn = gr.Button("Segment", variant="primary")
            with gr.Column():
                over = gr.Image(label="Overlay", format="jpeg")
                info = gr.Markdown()
        with gr.Row():
            mask = gr.Image(label="Mask / alpha", image_mode="L", format="png")
            cut = gr.Image(label="Cut-out", image_mode="RGBA", format="png")
        btn.click(run, [inp, tta, matting], [over, mask, cut, info])
        inp.upload(run, [inp, tta, matting], [over, mask, cut, info])
    return demo


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--share", action="store_true")
    p.add_argument("--port", type=int, default=7860)
    a = p.parse_args()
    get_segmenter()   # fail early if weights are missing
    build_demo().launch(server_name="0.0.0.0", server_port=a.port, share=a.share)
