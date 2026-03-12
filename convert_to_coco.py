import os
import json
import cv2
import glob
from tqdm import tqdm

def yolo_to_coco(root_dir, split):
    print(f"Converting {split} to COCO format...")
    img_dir = os.path.join(root_dir, split, 'images')
    lbl_dir = os.path.join(root_dir, split, 'labels')
    
    # Handle augmented data path
    if split == 'train':
        img_dir = os.path.join(root_dir, 'augmented_data', 'train', 'images')
        lbl_dir = os.path.join(root_dir, 'augmented_data', 'train', 'labels')
        
    class_names = ['plastic', 'metal', 'wood', 'glass', 'rubber', 'cloth', 'paper', 'fishing', 'bio', 'unknown']
    
    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i, "name": name} for i, name in enumerate(class_names)]
    }
    
    images = glob.glob(os.path.join(img_dir, '*.jpg'))
    ann_id = 0
    for i, img_path in enumerate(tqdm(images)):
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        img_id = i
        coco["images"].append({
            "id": img_id,
            "file_name": os.path.basename(img_path),
            "width": w,
            "height": h
        })
        
        name = os.path.splitext(os.path.basename(img_path))[0]
        l_path = os.path.join(lbl_dir, name + '.txt')
        if os.path.exists(l_path):
            with open(l_path, 'r') as f:
                for line in f:
                    p = line.strip().split()
                    if len(p) < 5: 
                        print(f"Skipping malformed line in {l_path}: {line}")
                        continue
                    c, cx, cy, bw, bh = int(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4])
                    # YOLO -> COCO [x_min, y_min, width, height]
                    abs_w = bw * w
                    abs_h = bh * h
                    x_min = (cx * w) - (abs_w / 2)
                    y_min = (cy * h) - (abs_h / 2)
                    
                    coco["annotations"].append({
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": c,
                        "bbox": [x_min, y_min, abs_w, abs_h],
                        "area": abs_w * abs_h,
                        "iscrowd": 0
                    })
                    ann_id += 1
                    
    out_dir = os.path.join(root_dir, 'coco_annotations')
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"instances_{split}.json"), 'w') as f:
        json.dump(coco, f)

if __name__ == "__main__":
    # Dynamically detect the project root
    rd = os.path.dirname(os.path.abspath(__file__))
    for s in ['train', 'val', 'test']:
        yolo_to_coco(rd, s)
