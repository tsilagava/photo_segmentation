"""Download the trained weights into weights/ (only needed if you got the code without them, e.g. as a source zip).

    python scripts/download_weights.py                         # uses the URLs below
    python scripts/download_weights.py --v5 <url> --v3 <url>
"""
import argparse
import urllib.request
from pathlib import Path

URLS = {
    "v5": "https://github.com/tsilagava/photo_segmentation/raw/main/weights/v5.pt",
    "v3": "https://github.com/tsilagava/photo_segmentation/raw/main/weights/v3.pt",
}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in URLS:
        p.add_argument(f"--{name}", default=URLS[name])
    p.add_argument("--out", default="weights")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in URLS:
        url, dest = getattr(a, name), out / f"{name}.pt"
        if not url:
            raise SystemExit(f"no URL for {name}: edit URLS in {__file__} or pass --{name} <url>")
        if dest.exists():
            print(dest, "already there")
            continue
        print("downloading", url)
        urllib.request.urlretrieve(url, dest)
        print(f"{dest}: {dest.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
