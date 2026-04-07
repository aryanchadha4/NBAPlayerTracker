import os
import csv
import json
import cv2


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


def save_manifest(data_dir, manifest_path):
    pairs = find_sample_pairs(data_dir)
    pairs.sort()
    with open(manifest_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_path', 'mask_path'])
        writer.writerows(pairs)
    return len(pairs)


def load_manifest(manifest_path):
    pairs = []
    with open(manifest_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pairs.append((row['image_path'], row['mask_path']))
    return pairs


def display_sample(pair):
    img_path, mask_path = pair
    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    print(f'Image: {img_path}')
    print(f'Mask: {mask_path}')
    cv2.imshow('Image', img)
    cv2.imshow('Mask', mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    manifest_path = os.path.join(data_dir, 'dataset_manifest.csv')
    count = save_manifest(data_dir, manifest_path)
    print(f'Wrote manifest with {count} image-mask pairs to {manifest_path}')
