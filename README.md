# NBAPlayerTracker

A basketball player detection and tracking project using the DeepSportRadar Basketball Instants dataset.

## Current status

- Dataset downloaded and extracted under `data/`
- Initial training script created in `scripts/train_segmentation.py`
- Basic dataset inspection notebook created at `notebooks/experiment_segmentation.ipynb`
- Environment prepared with `opencv-python`, `torch`, `torchvision`, `pycocotools`, `matplotlib`, and `numpy`

## Organization

- `data/`: dataset files and extracted images
- `data/dataset_manifest.csv`: generated image/mask manifest for training
- `models/`: saved model checkpoints
- `scripts/`: Python scripts for data loading, preprocessing, and training
- `notebooks/`: interactive experiment notebooks

## Next steps

1. Implement a COCO-style loader for detection and segmentation models.
2. Fine-tune a stronger model such as Mask R-CNN or YOLOv8.
3. Add multi-object tracking support using a tracker like DeepSORT.
4. Build a video inference pipeline that outputs player trajectories.

## Quick start

```bash
cd /Users/aryanchadha/NBAPlayerTracker
python3 scripts/train_segmentation.py
```
