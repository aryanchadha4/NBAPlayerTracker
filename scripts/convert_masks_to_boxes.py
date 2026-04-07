import os
import csv
import cv2
import numpy as np
import random

def mask_to_bbox(mask):
    """Convert binary mask to bounding box (x, y, w, h)"""
    if mask.sum() == 0:
        return None
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    return x_min, y_min, x_max - x_min, y_max - y_min

def create_yolo_dataset(manifest_path, output_dir, val_split=0.2):
    os.makedirs(os.path.join(output_dir, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images', 'val'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels', 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels', 'val'), exist_ok=True)

    samples = []
    with open(manifest_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append((row['image_path'], row['mask_path']))

    random.shuffle(samples)
    val_size = int(len(samples) * val_split)
    val_samples = samples[:val_size]
    train_samples = samples[val_size:]

    for split, sample_list in [('train', train_samples), ('val', val_samples)]:
        for img_path, mask_path in sample_list:
            # Copy image
            img_name = os.path.basename(img_path)
            img_dest = os.path.join(output_dir, 'images', split, img_name)
            os.system(f'cp "{img_path}" "{img_dest}"')

            # Create label
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            mask = (mask > 0).astype(np.uint8)
            bbox = mask_to_bbox(mask)
            if bbox is None:
                continue

            img = cv2.imread(img_path)
            h, w = img.shape[:2]
            x_min, y_min, bw, bh = bbox
            x_center = (x_min + bw / 2) / w
            y_center = (y_min + bh / 2) / h
            width = bw / w
            height = bh / h

            label_path = os.path.join(output_dir, 'labels', split, img_name.replace('.png', '.txt'))
            with open(label_path, 'w') as f:
                f.write(f'0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n')

if __name__ == '__main__':
    create_yolo_dataset('data/dataset_manifest.csv', 'data/yolo_data')