"""Train the U-Net / ResNet-34 pet segmenter (the v5 recipe from notebooks/Pet_Segmentation_Training_v5.ipynb).

    python train.py --data data/oxford-pets --out checkpoints/v5                 # v5: 384 px, small-pet oversampling
    python train.py --data data/oxford-pets --out checkpoints/v3 --img-size 256 --rotate-deg 0 \
        --min-crop-scale 0.5 --oversample-small 1                                  # roughly v3

The dataset is downloaded on first run. Checkpoints (best.pt, last.pt) and history.csv go to --out; re-running the
same command resumes from last.pt. When training ends the best checkpoint is scored on the test set (evaluate.py).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm

from petseg.data import (BG, PetData, PetSegDataset, eval_transform, label_lut, make_splits, seed_everything,
                         train_transform)
from petseg.metrics import CEDiceLoss, confusion, pet_iou, scores
from petseg.model import UNetResNet34


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/oxford-pets", help="dataset folder (downloaded if missing)")
    p.add_argument("--labels", default=None, help="trimap folder to train on (e.g. the audit's trimaps_clean/); default: original trimaps")
    p.add_argument("--splits", default=None, help="folder with train.txt / val.txt / test.txt (e.g. the notebooks' clean_splits/)")
    p.add_argument("--out", default="checkpoints/v5")
    p.add_argument("--img-size", type=int, default=384)
    p.add_argument("--rotate-deg", type=float, default=15)
    p.add_argument("--min-crop-scale", type=float, default=0.3)
    p.add_argument("--oversample-small", type=float, default=4.0, help="sampling weight of images whose pet covers <10%% (10–30%%: 1.5)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dice-weight", type=float, default=1.0)
    p.add_argument("--boundary-mode", choices=["ignore", "class"], default="ignore")
    p.add_argument("--patience", type=int, default=7, help="early stopping on validation mIoU")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=None, help="use only this many images per split (quick test runs)")
    p.add_argument("--device", default=None, help="cuda / mps / cpu (default: cuda if available)")
    p.add_argument("--no-eval", action="store_true", help="skip the test-set evaluation at the end")
    return p.parse_args()


def main():
    a = parse_args()
    assert a.img_size % 32 == 0, "--img-size must be divisible by 32"
    config = {"img_size": a.img_size, "rotate_deg": a.rotate_deg, "min_crop_scale": a.min_crop_scale,
              "oversample_small": a.oversample_small, "batch_size": a.batch_size, "epochs": a.epochs, "lr": a.lr,
              "weight_decay": a.weight_decay, "boundary_mode": a.boundary_mode, "dice_weight": a.dice_weight,
              "patience": a.patience, "seed": a.seed, "pretrained": True}
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print("device:", device)
    seed_everything(a.seed)
    torch.backends.cudnn.benchmark = True

    data = PetData(a.data, a.labels)
    data.download()
    splits = make_splits(data, a.splits, seed=a.seed)
    if a.limit:
        splits = {s: v[:a.limit] for s, v in splits.items()}
    for s, v in splits.items():
        print(f"{s:5s}: {len(v)} images")

    class_names = ["background", "pet"] + (["boundary"] if a.boundary_mode == "class" else [])
    n_cls = len(class_names)
    lut = label_lut(a.boundary_mode)
    train_ds = PetSegDataset(data, splits["train"], train_transform(a.img_size, a.rotate_deg, a.min_crop_scale), lut)
    val_ds = PetSegDataset(data, splits["val"], eval_transform(a.img_size), lut)

    # small-pet oversampling: pets under 10% of the frame drawn oversample_small x as often, 10–30% 1.5x
    shares = np.array([(data.trimap(n) != BG).mean() for n in tqdm(splits["train"], desc="pet shares")])
    w = np.where(shares < 0.10, a.oversample_small, np.where(shares < 0.30, 1.5, 1.0))
    sampler = WeightedRandomSampler(torch.tensor(w, dtype=torch.double), num_samples=len(w), replacement=True)
    kw = dict(batch_size=a.batch_size, num_workers=a.num_workers, pin_memory=device.type == "cuda",
              persistent_workers=a.num_workers > 0)
    train_dl = DataLoader(train_ds, sampler=sampler, drop_last=len(train_ds) > a.batch_size, **kw)
    val_dl = DataLoader(val_ds, shuffle=False, **kw)

    model = UNetResNet34(n_cls, pretrained=True).to(device).to(memory_format=torch.channels_last)
    criterion = CEDiceLoss(dice_weight=a.dice_weight)
    use_amp = device.type == "cuda"
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=a.lr, pct_start=0.1,
                                                    total_steps=a.epochs * len(train_dl))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    def run_epoch(loader, train):
        model.train(train)
        total, seen = 0.0, 0
        cm = torch.zeros(n_cls, n_cls, dtype=torch.long, device=device)
        ious = []
        with torch.set_grad_enabled(train):
            for x, y, _ in tqdm(loader, leave=False, desc="train" if train else "val"):
                x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
                y = y.to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    logits = model(x)
                    loss = criterion(logits, y)
                if train:
                    optimizer.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                pred = logits.argmax(1)
                cm += confusion(pred, y, n_cls)
                ious.append(pet_iou(pred, y))
                total += loss.item() * len(x)
                seen += len(x)
        return {"loss": total / seen, **scores(cm, class_names), "pet_iou": torch.cat(ious).mean().item()}

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    best_path, last_path = out / "best.pt", out / "last.pt"
    history, best_miou, bad, start = [], -1.0, 0, 1
    if last_path.exists():
        last = torch.load(last_path, map_location=device, weights_only=False)
        if last["config"] == config:
            model.load_state_dict(last["model"])
            optimizer.load_state_dict(last["optimizer"])
            scheduler.load_state_dict(last["scheduler"])
            scaler.load_state_dict(last["scaler"])
            history, best_miou, bad, start = last["history"], last["best_miou"], last["bad_epochs"], last["epoch"] + 1
            print(f"resuming after epoch {last['epoch']} (best val mIoU {best_miou:.4f})")
        else:
            print("last.pt was made with different settings, starting from scratch")
        del last

    for epoch in range(start, a.epochs + 1):
        if history and bad >= a.patience:
            print("early stopping")
            break
        t0 = time.time()
        tr, va = run_epoch(train_dl, True), run_epoch(val_dl, False)
        history.append({"epoch": epoch, "lr": scheduler.get_last_lr()[0],
                        **{f"train_{k}": v for k, v in tr.items()}, **{f"val_{k}": v for k, v in va.items()}})
        improved = va["miou"] > best_miou
        if improved:
            best_miou, bad = va["miou"], 0
            torch.save({"model": model.state_dict(), "config": config, "class_names": class_names, "epoch": epoch}, best_path)
        else:
            bad += 1
        print(f"epoch {epoch:3d} | {time.time() - t0:5.0f}s | train loss {tr['loss']:.4f} mIoU {tr['miou']:.4f} | "
              f"val loss {va['loss']:.4f} mIoU {va['miou']:.4f} pet IoU {va['pet_iou']:.4f}" + ("  *best*" if improved else ""))
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(), "config": config, "history": history, "best_miou": best_miou,
                    "bad_epochs": bad, "epoch": epoch}, last_path)
        pd.DataFrame(history).to_csv(out / "history.csv", index=False)

    (out / "splits.json").write_text(json.dumps(splits))
    print("best val mIoU:", round(best_miou, 4), "| checkpoint:", best_path)
    if not a.no_eval:
        from evaluate import evaluate
        evaluate({out.name: best_path}, data, splits["test"], out, device=device)


if __name__ == "__main__":
    main()
