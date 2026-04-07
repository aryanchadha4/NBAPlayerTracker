import argparse
from typing import Any

import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO


def show_results(results: Any) -> None:
    for result in results:
        image = result.plot()
        plt.figure(figsize=(12, 7))
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.show()


def run_inference(
    model_path: str,
    source: str,
    track: bool = False,
    tracker: str = "bytetrack.yaml",
    conf: float = 0.2,
    iou: float = 0.5,
):
    model = YOLO(model_path)
    if track:
        results = model.track(source=source, tracker=tracker, persist=True, conf=conf, iou=iou)
    else:
        results = model.predict(source=source, conf=conf, iou=iou)
    show_results(results)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLOv8 inference or tracking.")
    parser.add_argument("--model-path", default="runs/detect/player_tracking/weights/best.pt")
    parser.add_argument("--source", required=True)
    parser.add_argument("--track", action="store_true")
    parser.add_argument("--tracker", default="bytetrack.yaml")
    parser.add_argument("--conf", type=float, default=0.2)
    parser.add_argument("--iou", type=float, default=0.5)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_inference(
        model_path=args.model_path,
        source=args.source,
        track=args.track,
        tracker=args.tracker,
        conf=args.conf,
        iou=args.iou,
    )
