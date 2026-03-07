import cv2
import os
import glob
import random

def visualize():
    # Paths
    root_dir = r"c:\Users\palan\OneDrive\Desktop\Projects\Underwater Trash Detection Project"
    split = 'train'
    img_dir = os.path.join(root_dir, split, 'images')
    lbl_dir = os.path.join(root_dir, split, 'labels')
    
    # Class names mapping
    class_names = ['plastic', 'metal', 'wood', 'glass', 'rubber', 'cloth', 'paper', 'fishing', 'bio', 'unknown']
    
    # Get random images
    images = glob.glob(os.path.join(img_dir, '*.jpg'))
    if not images:
        print(f"No images found in {img_dir}")
        return
        
    num_samples = min(len(images), 5)
    samples = random.sample(images, num_samples)
    
    # Create output directory
    out_dir = os.path.join(root_dir, 'runs', 'detect', 'visualize_labels')
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Visualizing {num_samples} random images...")
    
    for img_path in samples:
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w, _ = img.shape
        
        # Corresponding label file
        bname = os.path.basename(img_path)
        name_no_ext = os.path.splitext(bname)[0]
        lbl_path = os.path.join(lbl_dir, f"{name_no_ext}.txt")
        
        if os.path.exists(lbl_path):
            with open(lbl_path, 'r') as f:
                for line in f:
                    parts = line.split()
                    if not parts: continue
                    cls_id = int(parts[0])
                    x, y, bw, bh = map(float, parts[1:])
                    
                    # De-normalize coordinates
                    x1 = int((x - bw/2) * w)
                    y1 = int((y - bh/2) * h)
                    x2 = int((x + bw/2) * w)
                    y2 = int((y + bh/2) * h)
                    
                    # Draw box and label
                    label = class_names[cls_id] if cls_id < len(class_names) else f"id_{cls_id}"
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Save output
        cv2.imwrite(os.path.join(out_dir, bname), img)

    print(f"Visualizations saved to: {out_dir}")

if __name__ == "__main__":
    visualize()
