import os
import csv
import json


def find_sample_pairs(data_dir):
    pairs = []
    for arena in os.listdir(data_dir):
        arena_path = os.path.join(data_dir, arena)
        if not os.path.isdir(arena_path) or not arena.startswith('KS-FR'):
            continue
        for game_id in os.listdir(arena_path):
            game_path = os.path.join(arena_path, game_id)
            if not os.path.isdir(game_path):
                continue
            for file_name in os.listdir(game_path):
                if file_name.endswith('_0.png'):
                    img_path = os.path.join(game_path, file_name)
                    mask_path = os.path.join(game_path, file_name.replace('_0.png', '_humans.png'))
                    if os.path.exists(mask_path):
                        pairs.append((img_path, mask_path))
    return pairs


def write_manifest(data_dir, manifest_path='dataset_manifest.csv'):
    pairs = find_sample_pairs(data_dir)
    pairs.sort()
    with open(manifest_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_path', 'mask_path'])
        for img_path, mask_path in pairs:
            writer.writerow([img_path, mask_path])
    return len(pairs)


if __name__ == '__main__':
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    manifest_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'dataset_manifest.csv')
    count = write_manifest(data_dir, manifest_path)
    print(f'Wrote dataset manifest with {count} image-mask pairs to {manifest_path}')
