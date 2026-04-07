import csv
import os
from typing import Dict, List


def _sample_record(data_dir: str, arena: str, game_id: str, file_name: str) -> Dict[str, str]:
    game_path = os.path.join(data_dir, arena, game_id)
    image_path = os.path.join(game_path, file_name)
    mask_path = os.path.join(game_path, file_name.replace("_0.png", "_humans.png"))
    annotation_path = os.path.join(game_path, file_name.replace("_0.png", ".json"))
    camera = file_name.split("_", 1)[0]

    return {
        "image_path": image_path,
        "mask_path": mask_path,
        "annotation_path": annotation_path,
        "arena": arena,
        "game_id": game_id,
        "camera": camera,
        "sequence_id": f"{arena}/{game_id}/{camera}",
    }


def find_sample_records(data_dir: str) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []
    for arena in os.listdir(data_dir):
        arena_path = os.path.join(data_dir, arena)
        if not os.path.isdir(arena_path) or not arena.startswith("KS-FR"):
            continue

        for game_id in os.listdir(arena_path):
            game_path = os.path.join(arena_path, game_id)
            if not os.path.isdir(game_path):
                continue

            for file_name in os.listdir(game_path):
                if not file_name.endswith("_0.png"):
                    continue

                record = _sample_record(data_dir, arena, game_id, file_name)
                if os.path.exists(record["mask_path"]) and os.path.exists(record["annotation_path"]):
                    records.append(record)

    return records


def save_manifest(data_dir: str, manifest_path: str) -> int:
    records = find_sample_records(data_dir)
    records.sort(key=lambda record: record["image_path"])

    fieldnames = [
        "image_path",
        "mask_path",
        "annotation_path",
        "arena",
        "game_id",
        "camera",
        "sequence_id",
    ]
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    return len(records)


def load_manifest(manifest_path: str) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []
    with open(manifest_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_path = row["image_path"]
            annotation_path = row.get("annotation_path")
            if not annotation_path:
                annotation_path = image_path.replace("_0.png", ".json")

            arena = row.get("arena") or os.path.basename(os.path.dirname(os.path.dirname(image_path)))
            game_id = row.get("game_id") or os.path.basename(os.path.dirname(image_path))
            file_name = os.path.basename(image_path)
            camera = row.get("camera") or file_name.split("_", 1)[0]

            records.append(
                {
                    "image_path": image_path,
                    "mask_path": row["mask_path"],
                    "annotation_path": annotation_path,
                    "arena": arena,
                    "game_id": game_id,
                    "camera": camera,
                    "sequence_id": row.get("sequence_id") or f"{arena}/{game_id}/{camera}",
                }
            )

    return records


if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    manifest_path = os.path.join(data_dir, "dataset_manifest.csv")
    count = save_manifest(data_dir, manifest_path)
    print(f"Wrote manifest with {count} image-mask-annotation rows to {manifest_path}")
