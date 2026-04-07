import argparse
import json
import os

import torch
from ultralytics import YOLO


def ensure_data_yaml(data_dir: str) -> str:
    yaml_path = os.path.join(data_dir, "data.yaml")
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(
            f"Expected dataset config at {yaml_path}. Run scripts/convert_masks_to_boxes.py first."
        )
    return yaml_path


def train_yolov8(
    data_dir: str,
    model_name: str = "yolov8s.pt",
    epochs: int = 75,
    batch_size: int = 8,
    imgsz: int = 1280,
    device: str | None = None,
    project: str = "runs/detect",
    name: str = "player_tracking",
) -> None:
    yaml_path = ensure_data_yaml(data_dir)
    model = YOLO(model_name)

    if device is None:
        device = "0" if torch.cuda.is_available() else "cpu"

    results = model.train(
        data=yaml_path,
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        patience=30,
        device=device,
        project=project,
        name=name,
        pretrained=True,
        cache=False,
        plots=True,
        close_mosaic=10,
    )

    print(
        json.dumps(
            {
                "save_dir": str(results.save_dir),
                "best_weights": os.path.join(str(results.save_dir), "weights", "best.pt"),
                "data_yaml": yaml_path,
                "imgsz": imgsz,
                "model": model_name,
            },
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLOv8 for player detection.")
    parser.add_argument("--data-dir", default="data/yolo_data")
    parser.add_argument("--model", default="yolov8s.pt")
    parser.add_argument("--epochs", type=int, default=75)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--device", default=None)
    parser.add_argument("--project", default="runs/detect")
    parser.add_argument("--name", default="player_tracking")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_yolov8(
        data_dir=args.data_dir,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        imgsz=args.imgsz,
        device=args.device,
        project=args.project,
        name=args.name,
    )
