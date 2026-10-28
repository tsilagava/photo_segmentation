"""Score one model or the ensemble on the test set, at full image resolution, with or without TTA.

    python evaluate.py                                    # every weights/*.pt, ensembled, with TTA
    python evaluate.py --weights weights/v5.pt --no-tta
    python evaluate.py --labels data/oxford-pets/annotations/trimaps_clean --splits clean_splits

Pet IoU is pet vs background on non-boundary pixels (the trimap's boundary band is excluded, as in training),
reported pooled over the whole test set and as a per-image mean, by pet size and by species.
Writes test_scores.csv and test_per_image.csv to --out.
"""
import argparse
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from petseg.data import BG, BND, PET, SIZE_BINS, SIZE_LABELS, PetData, breed_of, make_splits, species_of
from petseg.predict import PetSegmenter, find_weights


def evaluate(weights, data, names, out_dir, device=None, tta=True):
    seg = PetSegmenter(weights, device=device, max_side=None)
    variants = list(seg.models) + (["ensemble"] if len(seg.models) > 1 else [])
    pooled = {v: [0, 0] for v in variants}
    rows = []
    for n in tqdm(names, desc="test" + (" + TTA" if tta else "")):
        img, t = data.rgb(n), data.trimap(n)
        probs = seg.model_probs(img, tta)
        if "ensemble" in variants:
            probs["ensemble"] = sum(probs[m] * w for m, w in seg.ensemble_weights.items())
        valid, gt = t != BND, (t == PET) & (t != BND)
        r = {"filename": n, "breed": breed_of(n), "species": species_of(n), "pet_share": (t != BG).mean()}
        for v in variants:
            p = (probs[v] > 0.5) & valid
            i, u = (p & gt).sum(), (p | gt).sum()
            pooled[v][0] += i
            pooled[v][1] += u
            r[v] = i / u if u else 1.0
        rows.append(r)
    per_image = pd.DataFrame(rows)
    per_image["size_bin"] = pd.cut(per_image.pet_share, SIZE_BINS, labels=SIZE_LABELS, include_lowest=True)
    table = pd.DataFrame({v: {"pet IoU (whole test set)": pooled[v][0] / max(pooled[v][1], 1),
                              "pet IoU (per-image mean)": per_image[v].mean(),
                              **{f"pet IoU {l}": per_image.loc[per_image.size_bin == l, v].mean() for l in SIZE_LABELS},
                              **{f"pet IoU {s}s": per_image.loc[per_image.species == s, v].mean() for s in ("cat", "dog")},
                              "images": len(per_image)} for v in variants})
    table.columns = [f"{c} + TTA" if tta else c for c in table.columns]
    print(table.round(4).to_string())
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "test_scores.csv")
    per_image.to_csv(out_dir / "test_per_image.csv", index=False)
    print("hardest breeds:")
    print(per_image.groupby("breed")[variants].mean().sort_values(variants[-1]).head(6).round(4).to_string())
    return table


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", nargs="*", help="checkpoint files (default: every weights/*.pt)")
    p.add_argument("--data", default="data/oxford-pets")
    p.add_argument("--labels", default=None, help="trimaps to score against (default: original trimaps)")
    p.add_argument("--splits", default=None, help="folder with test.txt (default: official test list, empty masks dropped)")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--no-tta", action="store_true")
    p.add_argument("--device", default=None)
    p.add_argument("--out", default="results")
    a = p.parse_args()

    weights = {Path(w).stem: w for w in a.weights} if a.weights else find_weights()
    data = PetData(a.data, a.labels)
    data.download()
    names = make_splits(data, a.splits)["test"]
    if a.limit:
        names = names[:a.limit]
    evaluate(weights, data, names, a.out, device=a.device, tta=not a.no_tta)


if __name__ == "__main__":
    main()
