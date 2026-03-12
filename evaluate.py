from ultralytics import YOLO
import os

def evaluate_model():
    # Path to the best model weight
    # Dynamically detect the project root
    root_dir = os.path.dirname(os.path.abspath(__file__))
    # Use the provided weight in the root or the latest from runs
    model_path = os.path.join(root_dir, 'yolov8m.pt')
    if not os.path.exists(model_path):
        model_path = 'yolov8n.pt'
    
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
