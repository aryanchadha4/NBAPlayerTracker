import argparse
import json
import math
import os
import random
import shutil
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from data_loader import load_manifest


BBox = Tuple[int, int, int, int]


def clip_bbox(box: BBox, width: int, height: int) -> Optional[BBox]:
    x1, y1, x2, y2 = box
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width - 1))
    y2 = max(0, min(y2, height - 1))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def bbox_from_component(stats_row: np.ndarray, width: int, height: int) -> Optional[BBox]:
    x, y, w, h, area = [int(v) for v in stats_row]
    if area < 20 or w < 4 or h < 8:
        return None

    pad_x = max(4, int(round(w * 0.15)))
    pad_top = max(6, int(round(h * 0.20)))
    pad_bottom = max(2, int(round(h * 0.05)))
    return clip_bbox((x - pad_x, y - pad_top, x + w + pad_x, y + h + pad_bottom), width, height)


def heuristic_player_box(point: Sequence[float], width: int, height: int) -> Optional[BBox]:
    x, y = point
    if x < -0.1 * width or x > 1.1 * width or y < -0.1 * height or y > 1.1 * height:
        return None

    y_clamped = min(max(y, 0), height - 1)
    player_h = int(np.clip(0.05 * height + 0.08 * y_clamped, 28, 0.22 * height))
    player_w = int(np.clip(player_h * 0.42, 14, 0.09 * width))

    x1 = int(round(x - player_w / 2))
    x2 = int(round(x + player_w / 2))
    y2 = int(round(y + player_h * 0.06))
    y1 = int(round(y2 - player_h))
    return clip_bbox((x1, y1, x2, y2), width, height)


def extract_component_boxes(mask: np.ndarray) -> List[Tuple[BBox, Tuple[float, float]]]:
    binary = (mask > 0).astype(np.uint8)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    height, width = mask.shape[:2]

    components: List[Tuple[BBox, Tuple[float, float]]] = []
    for label_idx in range(1, num_labels):
        box = bbox_from_component(stats[label_idx], width, height)
        if box is None:
            continue
        x1, y1, x2, y2 = box
        anchor = ((x1 + x2) / 2.0, float(y2))
        components.append((box, anchor))
    return components


def distance_to_box_anchor(point: Sequence[float], anchor: Sequence[float]) -> float:
    return math.hypot(point[0] - anchor[0], point[1] - anchor[1])


def assign_component_box(
    point: Sequence[float],
    components: Sequence[Tuple[BBox, Tuple[float, float]]],
    used_indices: set,
) -> Optional[BBox]:
    candidates: List[Tuple[float, int, BBox]] = []
    x, y = point
    for idx, (box, anchor) in enumerate(components):
        if idx in used_indices:
            continue
        x1, y1, x2, y2 = box
        contains = x1 - 10 <= x <= x2 + 10 and y1 - 18 <= y <= y2 + 12
        dist = distance_to_box_anchor(point, anchor)
        if contains or dist < max(40.0, (y2 - y1) * 0.9):
            candidates.append((dist, idx, box))

    if not candidates:
        return None

    _, idx, box = min(candidates, key=lambda item: item[0])
    used_indices.add(idx)
    return box


def load_player_points(annotation_path: str) -> List[Tuple[float, float]]:
    with open(annotation_path) as f:
        data = json.load(f)

    points: List[Tuple[float, float]] = []
    for player in data.get("players", []):
        if player.get("status") != 1:
            continue
        pos_feet = player.get("pos_feet")
        if not pos_feet or len(pos_feet) != 2:
            continue
        points.append((float(pos_feet[0]), float(pos_feet[1])))
    return points


def dedupe_boxes(boxes: Iterable[BBox], iou_threshold: float = 0.85) -> List[BBox]:
    deduped: List[BBox] = []
    for box in boxes:
        x1, y1, x2, y2 = box
        area = max(0, x2 - x1) * max(0, y2 - y1)
        keep = True
        for other in deduped:
            ox1, oy1, ox2, oy2 = other
            inter_x1 = max(x1, ox1)
            inter_y1 = max(y1, oy1)
            inter_x2 = min(x2, ox2)
            inter_y2 = min(y2, oy2)
            inter = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
            union = area + max(0, ox2 - ox1) * max(0, oy2 - oy1) - inter
            iou = inter / union if union else 0.0
            if iou >= iou_threshold:
                keep = False
                break
        if keep:
            deduped.append(box)
    return deduped


def player_boxes_for_record(record: Dict[str, str]) -> List[BBox]:
    image = cv2.imread(record["image_path"])
    if image is None:
        raise FileNotFoundError(f"Could not read image: {record['image_path']}")

    height, width = image.shape[:2]
    mask = cv2.imread(record["mask_path"], cv2.IMREAD_GRAYSCALE)
    components = extract_component_boxes(mask) if mask is not None else []
    points = load_player_points(record["annotation_path"])

    boxes: List[BBox] = []
    used_indices: set = set()
    for point in points:
        component_box = assign_component_box(point, components, used_indices)
        box = component_box or heuristic_player_box(point, width, height)
        if box is not None:
            boxes.append(box)

    return dedupe_boxes(boxes)


def write_yolo_labels(label_path: str, boxes: Sequence[BBox], width: int, height: int) -> None:
    with open(label_path, "w") as f:
        for x1, y1, x2, y2 in boxes:
            bw = x2 - x1
            bh = y2 - y1
            x_center = (x1 + x2) / 2 / width
            y_center = (y1 + y2) / 2 / height
            f.write(f"0 {x_center:.6f} {y_center:.6f} {bw / width:.6f} {bh / height:.6f}\n")


def sequence_split(records: Sequence[Dict[str, str]], val_ratio: float, seed: int) -> Dict[str, str]:
    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for record in records:
        grouped[record["sequence_id"]].append(record)

    sequence_ids = list(grouped.keys())
    rng = random.Random(seed)
    rng.shuffle(sequence_ids)

    target_val = max(1, int(round(len(records) * val_ratio)))
    val_sequences = set()
    val_count = 0
    for sequence_id in sequence_ids:
        if val_count >= target_val:
            break
        val_sequences.add(sequence_id)
        val_count += len(grouped[sequence_id])

    return {
        record["image_path"]: ("val" if record["sequence_id"] in val_sequences else "train")
        for record in records
    }


def clean_output_dir(output_dir: str) -> None:
    for subdir in ["images/train", "images/val", "labels/train", "labels/val"]:
        path = os.path.join(output_dir, subdir)
        if os.path.isdir(path):
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)


def write_data_yaml(output_dir: str) -> None:
    yaml_path = os.path.join(output_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(
            "\n".join(
                [
                    f"path: {os.path.abspath(output_dir)}",
                    "train: images/train",
                    "val: images/val",
                    "",
                    "nc: 1",
                    "names: ['player']",
                    "",
                ]
            )
        )


def write_split_summary(output_dir: str, split_rows: Sequence[Tuple[str, str, str, int]]) -> None:
    summary_path = os.path.join(output_dir, "split_summary.csv")
    with open(summary_path, "w") as f:
        f.write("split,sequence_id,image_name,num_boxes\n")
        for split, sequence_id, image_name, num_boxes in split_rows:
            f.write(f"{split},{sequence_id},{image_name},{num_boxes}\n")


def create_yolo_dataset(manifest_path: str, output_dir: str, val_split: float = 0.2, seed: int = 42) -> Dict[str, int]:
    records = load_manifest(manifest_path)
    split_by_image = sequence_split(records, val_split, seed)

    os.makedirs(output_dir, exist_ok=True)
    clean_output_dir(output_dir)
    write_data_yaml(output_dir)

    counts = {"train_images": 0, "val_images": 0, "train_boxes": 0, "val_boxes": 0}
    split_rows: List[Tuple[str, str, str, int]] = []

    for record in records:
        split = split_by_image[record["image_path"]]
        image_name = os.path.basename(record["image_path"])
        image_dest = os.path.join(output_dir, "images", split, image_name)
        label_dest = os.path.join(output_dir, "labels", split, image_name.replace(".png", ".txt"))

        shutil.copy2(record["image_path"], image_dest)

        image = cv2.imread(record["image_path"])
        if image is None:
            raise FileNotFoundError(f"Could not read image: {record['image_path']}")

        height, width = image.shape[:2]
        boxes = player_boxes_for_record(record)
        write_yolo_labels(label_dest, boxes, width, height)

        counts[f"{split}_images"] += 1
        counts[f"{split}_boxes"] += len(boxes)
        split_rows.append((split, record["sequence_id"], image_name, len(boxes)))

    write_split_summary(output_dir, split_rows)
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a YOLO player-detection dataset.")
    parser.add_argument("--manifest", default="data/dataset_manifest.csv")
    parser.add_argument("--output", default="data/yolo_data")
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    stats = create_yolo_dataset(args.manifest, args.output, val_split=args.val_split, seed=args.seed)
    print(json.dumps(stats, indent=2))
