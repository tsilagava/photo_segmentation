# Pet segmentation: Oxford-IIIT Pet, end to end

Segments cats and dogs in photos: a **U-Net with an ImageNet-pretrained ResNet-34 encoder** trained on the
Oxford-IIIT Pet trimaps, two models (v3 at 256 px and v5 at 384 px) **ensembled with test-time augmentation**,
plus optional **soft edges with ViTMatte**. Includes training and evaluation scripts, a CLI, a REST API and a web demo.

Talent Growth Program · Machine Learning Engineer | Computer Vision · End-to-End ML Project

## შეფასების მასალები (deliverables)

| მოთხოვნა | სად არის |
|---|---|
| პროექტის სრული საწყისი კოდი | `petseg/` (მოდელი, მონაცემები, ინფერენსი), `train.py`, `evaluate.py`, `predict.py`, `api.py`, `app.py`, `notebooks/` |
| გაწვრთნისა და გამოყენების ინსტრუქცია | [Training](#training), [Using the model](#using-the-model) |
| ბიბლიოთეკები და დამოკიდებულებები | [`requirements.txt`](requirements.txt), [Installation](#installation) |
| სტრუქტურა და გაშვება | [Project structure](#project-structure), [Quick start](#quick-start) |
| ახალ გამოსახულებებზე გამოყენება | `predict.py`, ვებ დემო (`app.py`), API (`POST /predict`) |
| შედეგების ვიზუალური წარმოდგენა | overlay / mask / cut-out დემოში და `predict.py`-ის გამოსავალში, [Results](#results) |
| მოქმედი API / აპლიკაცია | `uvicorn api:app` → `/docs` (REST API) და `/ui` (ვებ დემო), Docker |
| გაწვრთნილი მოდელის ფაილი | [`weights/README.md`](weights/README.md): ჩამოტვირთვა, ექსპორტი ან ხელახლა გაწვრთნა |

## Quick start

```bash
git clone https://github.com/<user>/<repo>.git && cd <repo>
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_weights.py          # -> weights/v3.pt, weights/v5.pt
uvicorn api:app --port 8000                 # demo: http://localhost:8000/ui   API docs: http://localhost:8000/docs
```

## How it works

```
photo ──► v3 (256 px) + TTA ─┐
      └─► v5 (384 px) + TTA ─┴─► average pet probability ─► mask (p > 0.5)
                                                              │
                         optional: auto trimap (shrink / grow the mask) ─► ViTMatte ─► soft alpha matte
```

* **Model.** U-Net decoder on a ResNet-34 encoder (ImageNet weights), 24 M parameters, 2 classes (background, pet).
* **Labels.** Trimaps are 1 = pet, 2 = background, 3 = boundary. The boundary band is a wide strip of uncertain labels,
  so it is **ignored** in the loss and the metrics. The annotation audit (`notebooks/Pet_Annotation_Audit.ipynb`)
  produced cleaned trimaps: detached boundary strips became background, pet specks inside the band became boundary.
* **Clean split.** Official `trainval`/`test` lists, 80/20 breed-stratified train/val split (seed 42); images that don't
  decode, empty masks, masks a pretrained segmenter clearly disagrees with, and near-duplicates across splits are dropped
  (v4/v5 notebooks, section 2).
* **Training (v5).** 384 px, random rotation ±15°, random resized crops (30–100 %), flips, colour jitter;
  small pets (< 10 % of the frame) sampled 4× as often; cross-entropy + Dice, AdamW, one-cycle LR, mixed precision,
  early stopping on validation mIoU.
* **Inference.** Each model runs at its training size with **TTA** (horizontal flip × scales 0.75 / 1 / 1.25); the
  probabilities are resized to the photo and averaged over TTA, then over v3 and v5. v5 + TTA is best overall,
  v3 + TTA is better on small pets, and the ensemble combines both.
* **Soft edges.** A ring around the mask (2 % of the long side) is marked unknown and
  [ViTMatte](https://huggingface.co/hustvl/vitmatte-small-composition-1k) predicts the alpha there (fur, whiskers).

## Results

Official test set (empty masks dropped), full image resolution, cleaned trimaps, boundary excluded.
Numbers from `evaluate.py` / notebook section 4b (`comparison_full_test.csv`).

| Model | Pet IoU (whole test set) | Pet IoU (per image) | Pet IoU, pets < 10 % |
|---|---|---|---|
| v3 + TTA | TODO | TODO | TODO |
| v5 + TTA | TODO | TODO | TODO |
| **v3 + v5 ensemble** | TODO | TODO | TODO |

Reproduce with `python evaluate.py --labels <trimaps_clean> --splits <clean_splits>`.

## Installation

Python 3.10–3.12. A GPU is used if available, otherwise everything runs on CPU (inference ≈ 1–3 s per photo with TTA).

```bash
pip install -r requirements.txt
```

| Library | Used for |
|---|---|
| `torch`, `torchvision` | model, ResNet-34 weights, augmentation (`transforms.v2`) |
| `numpy`, `pillow`, `scipy` | image I/O, trimap morphology |
| `pandas`, `scikit-learn`, `tqdm` | splits, metrics tables, progress bars |
| `transformers` | ViTMatte soft edges (loaded only when matting is turned on) |
| `fastapi`, `uvicorn`, `python-multipart` | REST API |
| `gradio` | web demo |

## Getting the trained model

See [`weights/README.md`](weights/README.md). In short: `python scripts/download_weights.py`, or export your own
checkpoint with `python scripts/export_weights.py path/to/best.pt weights/v5.pt`, or retrain (below).

## Training

```bash
# v5 recipe (downloads Oxford-IIIT Pet, ~800 MB, into data/oxford-pets on first run)
python train.py --out checkpoints/v5
# v3-like model at 256 px
python train.py --out checkpoints/v3 --img-size 256 --rotate-deg 0 --min-crop-scale 0.5 --oversample-small 1
# export for inference
python scripts/export_weights.py checkpoints/v5/best.pt weights/v5.pt
python scripts/export_weights.py checkpoints/v3/best.pt weights/v3.pt
```

* `--labels` trains on the cleaned trimaps and `--splits` uses the clean split (both produced by the notebooks).
  Without them the original trimaps and the official split (empty masks dropped) are used.
* Re-running the same command resumes from `last.pt`. `history.csv` has per-epoch metrics.
* At the end the best checkpoint is scored on the test set (`test_scores.csv`, `test_per_image.csv`).
* A T4 GPU takes roughly 1–2 h for 40 epochs at 384 px. `--limit 32 --epochs 1` is a quick check that everything runs.

The original experiments are in `notebooks/` and run on Google Colab (T4, *Run all*): see [Notebooks](#notebooks).

## Using the model

**Web demo**

```bash
python app.py            # http://127.0.0.1:7860   (--share for a public link)
```

Upload a photo; you get the overlay, the mask and a transparent cut-out. Toggles for TTA and ViTMatte edges.

**REST API**

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

| Endpoint | |
|---|---|
| `GET /health` | loaded models and device |
| `POST /predict` | form field `file`; query `output=json\|mask\|overlay\|cutout`, `tta=true\|false`, `matting=true\|false` |
| `GET /docs` | interactive Swagger UI |
| `GET /ui` | the web demo |

```bash
curl -F file=@cat.jpg http://localhost:8000/predict
# {"width":500,"height":375,"pet_found":true,"pet_share":0.31,"bbox":[...],"models":["v3","v5"],...,"mask_png_base64":"..."}
curl -F file=@cat.jpg "http://localhost:8000/predict?output=cutout&matting=true" -o cat_cutout.png
```

**Command line**

```bash
python predict.py photo.jpg folder_of_photos/ --out predictions/ [--matting] [--no-tta]
```

writes `<name>_mask.png`, `<name>_overlay.jpg` and `<name>_cutout.png` per image.

**Python**

```python
from petseg import PetSegmenter, find_weights
seg = PetSegmenter(find_weights("weights"))
r = seg.predict("cat.jpg", tta=True, matting=False)   # r["mask"], r["prob"], r["alpha"], r["pet_share"]
```

**Docker**

```bash
docker build -t petseg .
docker run -p 8000:8000 -v "$PWD/weights:/app/weights" petseg
```

**Hugging Face Space (public demo).** Create a Gradio Space, push this repository to it together with `weights/*.pt`
(Spaces store large files with Git LFS); `app.py` is the entry point.

## Project structure

```
petseg/
  model.py          U-Net / ResNet-34, checkpoint loading
  data.py           dataset download, splits, labels, augmentation
  metrics.py        CE + Dice loss, IoU / Dice
  predict.py        PetSegmenter: TTA, ensemble, auto trimap, ViTMatte, overlay / cut-out
train.py            training (v5 recipe), resumable
evaluate.py         test-set evaluation (pooled / per-image IoU, by size, species, breed)
predict.py          batch prediction from the command line
api.py              FastAPI service (+ the demo at /ui)
app.py              Gradio web demo
scripts/            export_weights.py, download_weights.py
weights/            trained models (not in git; see weights/README.md)
notebooks/          the Colab experiments
tests/              smoke test (random weights, no dataset needed): pytest
Dockerfile
```

## Notebooks

| Notebook | What it does |
|---|---|
| `Pet_Annotation_Audit.ipynb` | audits the trimaps and writes the cleaned ones |
| `Pet_Segmentation_Training.ipynb`, `_v2` | first models; boundary as a third class, class-weighted loss |
| `Pet_Segmentation_Training_v3.ipynb` | boundary ignored, CE + Dice without class weights (256 px) |
| `Pet_Segmentation_Training_v4.ipynb` | clean split (duplicates, empty and suspicious masks removed), cleaned trimaps |
| `Pet_Segmentation_Training_v5.ipynb` | 384 px, small-pet oversampling, rotation, TTA; compares v3 / v4 / v5 |
| `Pet_Edges_v3_ViTMatte.ipynb` | auto trimap + ViTMatte soft edges on v3 |
| `Pet_Edges_Ensemble_ViTMatte.ipynb` | v3 + v5 ensemble with TTA + ViTMatte; full-test comparison |
| `notebook_agent.ipynb` | helper: a Gemini agent that drafted a training notebook from the EDA |

## Tests

```bash
pip install pytest httpx && pytest
```

## Data and licences

[Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/) (Parkhi et al., 2012), CC BY-SA 4.0.
ViTMatte: `hustvl/vitmatte-small-composition-1k` (Apache-2.0).
