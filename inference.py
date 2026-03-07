from ultralytics import YOLO
import os
import glob

def run_inference():
    # Path to the best model weight
    root_dir = r"c:\Users\palan\OneDrive\Desktop\Projects\Underwater Trash Detection Project"
    model_path = os.path.join(root_dir, 'runs', 'train', 'underwater_trash_yolov8n', 'weights', 'best.pt')
    
    if not os.path.exists(model_path):
        print(f"Model not found at: {model_path}")
        return

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
