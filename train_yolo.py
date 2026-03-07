from ultralytics import YOLO
import os

def train_yolo():
    print("--- PHASE 4A: YOLOv8 TRAINING ---")
    
    # Load pretrained YOLOv8m (medium)
    model = YOLO('yolov8m.pt')
    
    root_dir = r"c:\Users\palan\OneDrive\Desktop\Projects\Underwater Trash Detection Project"
    data_yaml = os.path.join(root_dir, 'data_aug.yaml')
    
    # Train with requested hyperparameters
    results = model.train(
        data=data_yaml,
        epochs=100,
        imgsz=640,
        batch=16,
        patience=20,
        optimizer='AdamW',
        lr0=0.001,
        augment=True,
        mosaic=1.0,
        mixup=0.1,
        cos_lr=True,
        plots=True,
        name='yolov8_underwater_final',
        project='runs/train',
        device='0' if torch.cuda.is_available() else 'cpu' 
    )
    
    print(f"YOLOv8 Training Complete. Best model saved at {results.save_dir}")

if __name__ == "__main__":
    train_yolo()
