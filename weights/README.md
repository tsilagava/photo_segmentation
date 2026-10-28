# Trained weights

Put the trained segmenters here as `v3.pt` and `v5.pt`. Every `*.pt` file in this folder is loaded and
the models are averaged (with one file, that model is used alone).

| File | Model | Input | Where it comes from |
|---|---|---|---|
| `v5.pt` | U-Net / ResNet-34, boundary ignored, small-pet oversampling | 384 px | `checkpoints_v5/best.pt` (Pet_Segmentation_Training_v5) |
| `v3.pt` | U-Net / ResNet-34, boundary ignored | 256 px | `checkpoints_v3/best.pt` (Pet_Segmentation_Training_v3) |

Three ways to get them:

1. **Download** the released files: `python scripts/download_weights.py`
2. **Export** your own checkpoints (e.g. from Google Drive, `upgaming/Animal Classification & Segmentation/checkpoints_v*/best.pt`):
   `python scripts/export_weights.py path/to/best.pt weights/v5.pt`
3. **Retrain** with `train.py` (see the main README) and export `checkpoints/v5/best.pt` the same way.

The `.pt` files are git-ignored; publish them as GitHub Release assets (or on the Hugging Face Hub) instead of committing them.
