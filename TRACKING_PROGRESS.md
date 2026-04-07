# Tracking Progress

Last updated: 2026-04-07

## Status

- Done: Rebuilt the manifest to include per-frame annotation JSON and sequence ids.
- Done: Replaced single crowd-box labels with per-player label generation in `scripts/convert_masks_to_boxes.py`.
- Done: Switched train/val splitting from random frame shuffle to deterministic sequence-level splitting.
- Done: Updated YOLO training defaults for small-player detection with a larger model and higher image size.
- Done: Added ByteTrack-enabled inference for detection-to-tracking evaluation.
- Done: Regenerated `data/yolo_data` with 250 train images, 74 val images, 2,469 train boxes, and 673 val boxes.
- Done: Verified there are no missing label files and no train/val sequence overlap.
- Blocked by data availability: Dataset size is still limited by the source annotations already in the repo.

## Success Criteria

- Average boxes per image is greater than 1 and matches visible players much more closely than before.
  Current result: `train=9.88`, `val=9.09`.
- No images are missing label files after dataset generation.
  Current result: `0` missing labels.
- Validation sequences do not overlap training sequences.
  Current result: `0` overlapping sequences.
- Tracking runs on top of detector outputs using `scripts/infer_yolov8.py --track`.
