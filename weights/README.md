# Trained weights

The two trained segmenters live here. Every `*.pt` file in this folder is loaded and the models are averaged
(with one file, that model is used alone).

| File | Model | Input | Epoch | SHA-256 |
|---|---|---|---|---|
| `v3.pt` | U-Net / ResNet-34, boundary ignored | 256 px | 34 | `d0014b17e4d35074493fc41ff938161dd417b5d46e60dd446c02b008a12fd94b` |
| `v5.pt` | U-Net / ResNet-34, boundary ignored, small-pet oversampling, rotation | 384 px | 32 | `e24a8909a5840a006c1ff5d017ee0c8b315ce5eb72fdfa0238ba2bc4b546db52` |

Each file is the `best.pt` checkpoint from `Pet_Segmentation_Training_v3` / `_v5` (best validation mIoU), stripped
to the model weights in float16 by `scripts/export_weights.py` (49 MB instead of 98 MB; loading converts back to float32).

To get them again:

1. **Clone** the repository: they come with it.
2. **Download** them on their own: `python scripts/download_weights.py`
3. **Export** a training checkpoint: `python scripts/export_weights.py path/to/best.pt weights/v5.pt`
4. **Retrain** with `train.py` (see the main README) and export `checkpoints/v5/best.pt` the same way.
