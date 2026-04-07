import cv2
import numpy as np
import os
import json

def preprocess_image_opencv(img, target_size=(640, 480)):
    resized = cv2.resize(img, target_size)
    normalized = resized / 255.0  # Normalize to [0,1]
    return normalized

def extract_centroids(annotations):
    centroids = []
    for ann in annotations:
        if ann['type'] == 'ball':
            # Ball has center [x,y,z], but for 2D, use x,y
            centroids.append((ann['center'][0], ann['center'][1]))
        # For players, if masks, but in this dataset, annotations are ball or humans?
        # From inspection, annotations are list of dicts with type, center, etc.
    return centroids

# Load from the custom format
def load_custom_dataset(data_dir):
    json_file = os.path.join(data_dir, 'basketball-instants-dataset.json')
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    images = []
    annotations = []
    for item in data:
        # Find image path, perhaps based on arena_label and game_id
        arena = item['arena_label']
        game_id = item['game_id']
        # Images are in folders like KS-FR-BLOIS/24330/
        # But need to find the exact path
        # For now, skip loading images, just process annotations
        anns = item['annotations']
        annotations.append(anns)
    
    return None, annotations  # Images not loaded yet

if __name__ == "__main__":
    data_dir = '/Users/aryanchadha/NBAPlayerTracker/data'
    _, anns = load_custom_dataset(data_dir)
    print(f"Loaded {len(anns)} annotation sets.")