from ultralytics import YOLO
import os

def train_model():
    # Load a pretrained YOLOv8n model
    model = YOLO('yolov8n.pt')

    # Project path
    root_dir = r"c:\Users\palan\OneDrive\Desktop\Projects\Underwater Trash Detection Project"
    data_yaml = os.path.join(root_dir, 'data.yaml')

    # Training parameters
    results = model.train(
        data=data_yaml,
        epochs=5,
        imgsz=480,
        batch=16,
        name='underwater_trash_yolov8n',
        project='runs/train',
        device='cpu'  # Default to CPU, user can change if GPU is available
    )

    print("Training completed.")
    print(f"Model saved at: {results.save_dir}")

if __name__ == "__main__":
    train_model()
