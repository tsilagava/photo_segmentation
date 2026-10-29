"""Oxford-IIIT Pet: download, splits, labels and the training augmentation (v5 recipe)."""
import random
import shutil
import tarfile
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2

ARCHIVES = {
    "images.tar.gz": "https://thor.robots.ox.ac.uk/pets/images.tar.gz",
    "annotations.tar.gz": "https://thor.robots.ox.ac.uk/pets/annotations.tar.gz",
}
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
IGNORE = 255
PET, BG, BND = 1, 2, 3   # trimap values
SIZE_BINS = [0, 0.10, 0.30, 0.50, 1.0]   # pet share of the image, for the size-grouped evaluation
SIZE_LABELS = ["<10%", "10–30%", "30–50%", ">50%"]


class PetData:
    """Paths of the extracted dataset. label_dir defaults to the original trimaps (use the audit's cleaned ones if you have them)."""

    def __init__(self, root, label_dir=None):
        self.root = Path(root)
        self.images = self.root / "images"
        self.ann = self.root / "annotations"
        self.labels = Path(label_dir) if label_dir else self.ann / "trimaps"

    def ready(self):
        return self.images.is_dir() and (self.ann / "trimaps").is_dir() and (self.ann / "trainval.txt").exists()

    def download(self):
        if self.ready():
            return
        self.root.mkdir(parents=True, exist_ok=True)
        for name, url in ARCHIVES.items():
            local = self.root / name
            if not local.exists():
                download(url, local)
            print("extracting", name)
            with tarfile.open(local) as t:
                try:
                    t.extractall(self.root, filter="data")
                except TypeError:  # Python without tarfile filters
                    t.extractall(self.root)
            local.unlink()
        assert self.ready(), f"dataset not found in {self.root}"

    def usable(self, name):
        return (not name.startswith("._") and (self.images / f"{name}.jpg").exists()
                and (self.labels / f"{name}.png").exists())

    def trimap(self, name):
        return np.array(Image.open(self.labels / f"{name}.png"))

    def rgb(self, name):
        return np.array(Image.open(self.images / f"{name}.jpg").convert("RGB"))


def download(url, dest, retries=10):
    """Downloads to dest.part, resuming after dropped connections, and renames to dest only when complete."""
    part = Path(f"{dest}.part")
    print("downloading", url)
    for attempt in range(retries):
        have = part.stat().st_size if part.exists() else 0
        req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                if have and r.status != 206:   # server ignored the range: start over
                    have = 0
                total = have + int(r.headers.get("Content-Length", 0))
                with open(part, "ab" if have else "wb") as f:
                    shutil.copyfileobj(r, f, 1 << 20)
            if part.stat().st_size >= total:
                part.rename(dest)
                return
        except OSError as e:
            print(f"  connection dropped at {(part.stat().st_size if part.exists() else 0) / 1e6:.0f} MB ({e}), resuming")
            time.sleep(min(30, 2 ** attempt))
    raise RuntimeError(f"could not download {url}")


def read_list(path):
    return [l.split()[0] for l in Path(path).read_text().splitlines() if l.strip() and not l.startswith("#")]


def breed_of(name):
    return name.rsplit("_", 1)[0]


def species_of(name):
    return "cat" if name[0].isupper() else "dog"


def make_splits(data, splits_dir=None, val_frac=0.20, seed=42):
    """{"train", "val", "test": names}.

    With splits_dir (train.txt / val.txt / test.txt, e.g. the notebooks' clean_splits/) those lists are used as-is.
    Otherwise: official trainval/test lists, an 80/20 breed-stratified train/val split (seed 42, as in the notebooks)
    and empty masks dropped. The near-duplicate and pretrained-segmenter screens live in Pet_Segmentation_Training_v4/v5.
    """
    if splits_dir:
        return {s: [n for n in read_list(Path(splits_dir) / f"{s}.txt") if data.usable(n)] for s in ("train", "val", "test")}
    from sklearn.model_selection import train_test_split

    def keep(names):
        return [n for n in names if data.usable(n) and (data.trimap(n) == PET).any()]

    trainval = keep(read_list(data.ann / "trainval.txt"))
    test = keep(read_list(data.ann / "test.txt"))
    tr, va = train_test_split(trainval, test_size=val_frac, random_state=seed, stratify=[breed_of(n) for n in trainval])
    return {"train": tr, "val": va, "test": test}


def label_lut(boundary_mode="ignore"):
    """trimap value -> label: 0 background, 1 pet, boundary either ignored (255) or class 2."""
    mapping = {PET: 1, BG: 0, BND: IGNORE if boundary_mode == "ignore" else 2}
    lut = np.full(256, IGNORE, dtype=np.uint8)
    for k, v in mapping.items():
        lut[k] = v
    return lut


def train_transform(size, rotate_deg=15, min_crop_scale=0.3):
    # same geometry for image and mask (nearest-neighbour for masks); colour jitter / normalisation touch the image only
    return v2.Compose([
        v2.RandomRotation(rotate_deg, fill={tv_tensors.Image: 0, tv_tensors.Mask: IGNORE}),
        v2.RandomResizedCrop((size, size), scale=(min_crop_scale, 1.0), ratio=(0.75, 1.33), antialias=True),
        v2.RandomHorizontalFlip(),
        v2.RandomApply([v2.ColorJitter(0.3, 0.3, 0.3, 0.05)], p=0.8),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(MEAN, STD),
    ])


def eval_transform(size):
    return v2.Compose([v2.Resize((size, size), antialias=True), v2.ToDtype(torch.float32, scale=True), v2.Normalize(MEAN, STD)])


class PetSegDataset(Dataset):
    def __init__(self, data, names, transform, lut):
        self.data, self.names, self.transform, self.lut = data, list(names), transform, lut

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        name = self.names[i]
        img = tv_tensors.Image(torch.from_numpy(self.data.rgb(name)).permute(2, 0, 1))
        mask = tv_tensors.Mask(torch.from_numpy(self.lut[self.data.trimap(name)]))
        img, mask = self.transform(img, mask)
        return img, mask.long(), i


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
