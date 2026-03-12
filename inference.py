from ultralytics import YOLO
import os
import glob

def run_inference():
    # Path to the best model weight
    # Dynamically detect the project root
    root_dir = os.path.dirname(os.path.abspath(__file__))
    # Use the provided weight in the root or a default
    model_path = os.path.join(root_dir, 'yolov8m.pt')
    if not os.path.exists(model_path):
        model_path = 'yolov8n.pt'
        
    model = YOLO(model_path)
    
    # Get 5 sample images from the test set
    test_images_dir = os.path.join(root_dir, 'test', 'images')
    samples = glob.glob(os.path.join(test_images_dir, '*.jpg'))[:5]
    
    if not samples:
        print(f"No test images found in {test_images_dir}")
        return

    # Run inference and save annotated images
    print(f"Running inference on {len(samples)} samples...")
    results = model.predict(
        source=samples, 
        save=True, 
        project=os.path.join(root_dir, 'runs/detect'), 
        name='inference_samples',
        exist_ok=True
    )
    
    print(f"Inference completed. Annotated images saved in: {results[0].save_dir}")

if __name__ == "__main__":
    run_inference()
