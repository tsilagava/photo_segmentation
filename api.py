"""REST API (FastAPI) with the Gradio demo mounted at /ui.

    uvicorn api:app --host 0.0.0.0 --port 8000
    # docs:  http://localhost:8000/docs      demo: http://localhost:8000/ui

    curl -F file=@cat.jpg "http://localhost:8000/predict"                          # JSON (stats + base64 mask)
    curl -F file=@cat.jpg "http://localhost:8000/predict?output=cutout" -o cut.png  # transparent PNG
"""
import base64
import io
import os
from contextlib import asynccontextmanager
from enum import Enum

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import RedirectResponse, Response
from PIL import Image, UnidentifiedImageError

from petseg.predict import PetSegmenter, cutout, find_weights, overlay

state = {}


@asynccontextmanager
async def lifespan(app):
    state["seg"] = PetSegmenter(find_weights(os.environ.get("WEIGHTS_DIR", "weights")))
    if demo_module:
        demo_module._segmenter = state["seg"]   # the /ui demo shares the loaded models
    yield
    state.clear()


app = FastAPI(title="Pet segmentation API", version="1.0",
              description="U-Net / ResNet-34 pet segmentation (Oxford-IIIT Pet), v3 + v5 ensemble with TTA and optional ViTMatte edges.",
              lifespan=lifespan)


class Output(str, Enum):
    json = "json"
    mask = "mask"
    overlay = "overlay"
    cutout = "cutout"


def png(arr):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/ui")


@app.get("/health")
def health():
    seg = state["seg"]
    return {"status": "ok", "models": {n: size for n, (_, size) in seg.models.items()}, "device": str(seg.device)}


@app.post("/predict")
async def predict(file: UploadFile = File(..., description="image (jpg / png / webp)"),
                  output: Output = Query(Output.json, description="json, mask (PNG), overlay (PNG) or cutout (RGBA PNG)"),
                  tta: bool = Query(True, description="test-time augmentation"),
                  matting: bool = Query(False, description="refine the edge with ViTMatte")):
    try:
        image = Image.open(io.BytesIO(await file.read()))
        image.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "could not read the image")
    r = state["seg"].predict(image, tta=tta, matting=matting)
    alpha8 = (r["alpha"] * 255).round().astype(np.uint8)
    if output == Output.mask:
        return Response(png(alpha8), media_type="image/png")
    if output == Output.overlay:
        return Response(png(overlay(r["image"], r["alpha"])), media_type="image/png")
    if output == Output.cutout:
        return Response(png(cutout(r["image"], r["alpha"])), media_type="image/png")
    ys, xs = np.nonzero(r["mask"])
    return {"width": r["image"].shape[1], "height": r["image"].shape[0], "pet_found": bool(len(xs)),
            "pet_share": round(r["pet_share"], 4),
            "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if len(xs) else None,
            "models": list(state["seg"].models), "tta": tta, "matting": matting,
            "mask_png_base64": base64.b64encode(png(alpha8)).decode()}


try:
    import gradio as gr

    import app as demo_module
    app = gr.mount_gradio_app(app, demo_module.build_demo(), path="/ui")
except ImportError:   # gradio not installed: API only
    demo_module = None
