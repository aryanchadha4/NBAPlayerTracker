import torch
from ultralytics import YOLO
import os

def train_yolov8(data_dir, epochs=50, batch_size=16):
    # Create data.yaml
    yaml_content = f"""
train: {data_dir}/images/train
val: {data_dir}/images/val

nc: 1
names: ['player']
"""
    yaml_path = os.path.join(data_dir, 'data.yaml')
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    # Load model
    model = YOLO('yolov8n.pt')  # nano model for speed

    # Train
    results = model.train(data=yaml_path, epochs=epochs, batch=batch_size, imgsz=640, device='cuda' if torch.cuda.is_available() else 'cpu')

    # Model is saved automatically to runs/detect/train/weights/best.pt
    print("Model trained and saved to runs/detect/train/weights/best.pt")

if __name__ == '__main__':
    train_yolov8('data/yolo_data')