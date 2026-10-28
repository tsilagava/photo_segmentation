"""End-to-end smoke test with randomly initialised weights (no dataset or trained model needed)."""
import io

import numpy as np
import pytest
import torch
from PIL import Image

from petseg.model import UNetResNet34
from petseg.predict import PetSegmenter, cutout, overlay


@pytest.fixture(scope="module")
def weights_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("weights")
    torch.manual_seed(0)
    for name, size in [("v3", 256), ("v5", 384)]:
        net = UNetResNet34(2, pretrained=False)
        torch.save({"model": {k: v.half() if v.is_floating_point() else v for k, v in net.state_dict().items()},
                    "config": {"img_size": size, "boundary_mode": "ignore"}, "class_names": ["background", "pet"],
                    "epoch": 1}, d / f"{name}.pt")
    return d


def test_predictor(weights_dir):
    seg = PetSegmenter({p.stem: p for p in weights_dir.glob("*.pt")}, device="cpu", max_side=300)
    img = (np.random.default_rng(0).random((480, 640, 3)) * 255).astype(np.uint8)
    r = seg.predict(img)
    assert r["image"].shape == (225, 300, 3)            # long side capped at max_side
    assert r["prob"].shape == r["mask"].shape == (225, 300)
    assert 0 <= r["prob"].min() and r["prob"].max() <= 1
    assert overlay(r["image"], r["alpha"]).shape == (225, 300, 3)
    assert cutout(r["image"], r["alpha"]).shape == (225, 300, 4)
    tri = seg.auto_trimap(np.pad(np.ones((50, 50), bool), 50))
    assert set(np.unique(tri)) == {0, 128, 255}


def test_api(weights_dir, monkeypatch):
    monkeypatch.setenv("WEIGHTS_DIR", str(weights_dir))
    from fastapi.testclient import TestClient

    import api
    buf = io.BytesIO()
    Image.fromarray((np.random.default_rng(1).random((200, 300, 3)) * 255).astype(np.uint8)).save(buf, format="JPEG")
    with TestClient(api.app) as client:
        assert client.get("/health").json()["models"] == {"v3": 256, "v5": 384}
        r = client.post("/predict", files={"file": ("x.jpg", buf.getvalue(), "image/jpeg")}, params={"tta": False})
        assert r.status_code == 200 and {"pet_share", "mask_png_base64"} <= r.json().keys()
        r = client.post("/predict", files={"file": ("x.jpg", buf.getvalue(), "image/jpeg")}, params={"output": "cutout"})
        assert Image.open(io.BytesIO(r.content)).mode == "RGBA"
        assert client.post("/predict", files={"file": ("x.txt", b"not an image", "text/plain")}).status_code == 400
        assert client.get("/ui/").status_code == 200
