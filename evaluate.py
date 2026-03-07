from ultralytics import YOLO
import os

def evaluate_model():
    # Path to the best model weight
    root_dir = r"c:\Users\palan\OneDrive\Desktop\Projects\Underwater Trash Detection Project"
    model_path = os.path.join(root_dir, 'runs', 'train', 'underwater_trash_yolov8n', 'weights', 'best.pt')
    
    if not os.path.exists(model_path):
        print(f"Model not found at: {model_path}")
        # Try finding it in the project directory if the run name was different
        print("Checking alternative paths...")
        # For simplicity, if not found, exit
        return

    model = YOLO(model_path)
    
    # Evaluate on the test split
    results = model.val(data=os.path.join(root_dir, 'data.yaml'), split='test')
    
    print("\n" + "="*30)
    print("Evaluation Results on Test Set")
    print("="*30)
    print(f"mAP50: {results.box.map50:.4f}")
    print(f"mAP50-95: {results.box.map:.4f}")
    print("="*30)

if __name__ == "__main__":
    evaluate_model()
