import os
import cv2
import hashlib
import pandas as pd
from PIL import Image
from tqdm import tqdm
from collections import Counter

def get_file_hash(file_path):
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def analyze_dataset(root_dir):
    report = []
    summary_data = {
        'split': [],
        'num_images': [],
        'num_labels': [],
        'missing_labels': [],
        'orphan_labels': [],
        'corrupt_images': [],
        'duplicates': []
    }
    
    class_counts = Counter()
    widths, heights = [], []
    all_hashes = {}
    
    splits = ['train', 'val', 'test']
    
    for split in splits:
        split_path = os.path.join(root_dir, split)
        if not os.path.exists(split_path):
            continue
            
        # Standardize paths for this recon based on standard YOLO if already restructured, 
        # but the request asks to traverse subfolders.
        img_dir = os.path.join(split_path, 'images')
        lbl_dir = os.path.join(split_path, 'labels')
        
        # If not restructured yet, look in split root
        if not os.path.exists(img_dir): img_dir = split_path
        if not os.path.exists(lbl_dir): lbl_dir = split_path
        
        images = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        labels = [f for f in os.listdir(lbl_dir) if f.lower().endswith('.txt')]
        
        img_stems = {os.path.splitext(f)[0] for f in images}
        lbl_stems = {os.path.splitext(f)[0] for f in labels}
        
        missing = img_stems - lbl_stems
        orphans = lbl_stems - img_stems
        
        corrupt = 0
        dupes = 0
        
        for img_name in images:
            img_path = os.path.join(img_dir, img_name)
            
            # Check for corruption
            try:
                with Image.open(img_path) as img:
                    img.verify()
                # Re-open for size
                with Image.open(img_path) as img:
                    w, h = img.size
                    widths.append(w)
                    heights.append(h)
            except:
                corrupt += 1
                continue
            
            # Check for duplicates
            f_hash = get_file_hash(img_path)
            if f_hash in all_hashes:
                dupes += 1
            else:
                all_hashes[f_hash] = img_path
                
            # Class distribution from labels
            name_no_ext = os.path.splitext(img_name)[0]
            l_path = os.path.join(lbl_dir, f"{name_no_ext}.txt")
            if os.path.exists(l_path):
                with open(l_path, 'r') as f:
                    for line in f:
                        parts = line.split()
                        if parts:
                            class_counts[parts[0]] += 1
                            
        summary_data['split'].append(split)
        summary_data['num_images'].append(len(images))
        summary_data['num_labels'].append(len(labels))
        summary_data['missing_labels'].append(len(missing))
        summary_data['orphan_labels'].append(len(orphans))
        summary_data['corrupt_images'].append(corrupt)
        summary_data['duplicates'].append(dupes)

    df_summary = pd.DataFrame(summary_data)
    
    print("\n--- PHASE 0: DATASET SUMMARY REPORT ---")
    header = summary_data.keys()
    print(" | ".join(header))
    print("-" * 100)
    for i in range(len(summary_data['split'])):
        row = [str(summary_data[k][i]) for k in header]
        print(" | ".join(row))
    
    if widths:
        print("\n--- IMAGE DIMENSIONS ---")
        print(f"Min: {min(widths)}x{min(heights)}")
        print(f"Max: {max(widths)}x{max(heights)}")
        print(f"Mean: {sum(widths)//len(widths)}x{sum(heights)//len(heights)}")
        
    print("\n--- CLASS DISTRIBUTION (FROM YOLO LABELS) ---")
    # Mapping back to names if possible
    class_names = ['plastic', 'metal', 'wood', 'glass', 'rubber', 'cloth', 'paper', 'fishing', 'bio', 'unknown']
    for cls_id, count in class_counts.items():
        name = class_names[int(cls_id)] if int(cls_id) < len(class_names) else f"ID_{cls_id}"
        print(f"{name}: {count}")

if __name__ == "__main__":
    # Dynamically detect the project root
    root_dir = os.path.dirname(os.path.abspath(__file__))
    analyze_dataset(root_dir)
