from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

def infer_yolov8(model_path='runs/detect/train/weights/best.pt', image_path=None):
    model = YOLO(model_path)
    results = model(image_path)
    
    # Display results
    for result in results:
        img = result.plot()
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()
    
    return results

if __name__ == '__main__':
    # Example inference on a sample image
    sample_img = 'data/KS-FR-FOS/24335/camcourt1_1513710233743_0.png'  # Adjust path
    results = infer_yolov8(image_path=sample_img)