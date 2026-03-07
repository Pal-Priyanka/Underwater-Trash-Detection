import os
import cv2
import cv2
import albumentations as A
import glob
import numpy as np
from tqdm import tqdm
from collections import Counter

def run_augmentation(root_dir):
    print("--- PHASE 3: FEATURE ENGINEERING & AUGMENTATION ---")
    
    class_names = ['plastic', 'metal', 'wood', 'glass', 'rubber', 'cloth', 'paper', 'fishing', 'bio', 'unknown']
    train_img_dir = os.path.join(root_dir, 'train', 'images')
    train_lbl_dir = os.path.join(root_dir, 'train', 'labels')
    
    # 1. Check imbalance
    label_files = glob.glob(os.path.join(train_lbl_dir, '*.txt'))
    class_counts = Counter()
    img_to_classes = {}
    
    for lp in label_files:
        with open(lp, 'r') as f:
            classes = set()
            for line in f:
                parts = line.strip().split()
                if parts:
                    cls_id = int(parts[0])
                    class_counts[cls_id] += 1
                    classes.add(cls_id)
            img_to_classes[os.path.basename(lp)] = classes
            
    print(f"Initial counts: {dict(class_counts)}")
    
    # Define augmentations
    aug_pipeline = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.4),
        A.HueSaturationValue(p=0.3),
        A.GaussianBlur(p=0.2),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.3),
        A.CLAHE(p=0.3)
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

    # Output directory for augmented data
    aug_root = os.path.join(root_dir, 'augmented_data')
    aug_img_dir = os.path.join(aug_root, 'train', 'images')
    aug_lbl_dir = os.path.join(aug_root, 'train', 'labels')
    os.makedirs(aug_img_dir, exist_ok=True)
    os.makedirs(aug_lbl_dir, exist_ok=True)
    
    # 2. Process all original images first (resize 640x640 for YOLO default)
    print("Processing original images...")
    for img_name in tqdm(os.listdir(train_img_dir)):
        if not img_name.lower().endswith('.jpg'): continue
        shutil.copy(os.path.join(train_img_dir, img_name), os.path.join(aug_img_dir, img_name))
        l_name = os.path.splitext(img_name)[0] + '.txt'
        if os.path.exists(os.path.join(train_lbl_dir, l_name)):
            shutil.copy(os.path.join(train_lbl_dir, l_name), os.path.join(aug_lbl_dir, l_name))

    # 3. Oversampling for minority classes
    # Target: Ratio <= 3:1 relative to max class
    max_count = max(class_counts.values())
    target_count = max_count // 3
    
    print(f"Target count per class: {target_count}")
    
    for cls_id in range(len(class_names)):
        if class_counts[cls_id] < target_count and class_counts[cls_id] > 0:
            deficit = target_count - class_counts[cls_id]
            # Find images containing this class
            source_images = [k for k, v in img_to_classes.items() if cls_id in v]
            if not source_images: continue
            
            print(f"Oversampling class {class_names[cls_id]} by {deficit} samples...")
            for i in range(deficit):
                src_lbl_name = np.random.choice(source_images)
                name_no_ext = os.path.splitext(src_lbl_name)[0]
                img_path = os.path.join(train_img_dir, name_no_ext + '.jpg')
                lbl_path = os.path.join(train_lbl_dir, src_lbl_name)
                
                image = cv2.imread(img_path)
                bboxes = []
                with open(lbl_path, 'r') as f:
                    for line in f:
                        p = line.split()
                        bboxes.append([float(p[1]), float(p[2]), float(p[3]), float(p[4]), int(p[0])])
                
                # Apply augmentation
                try:
                    augmented = aug_pipeline(image=image, bboxes=[b[:4] for b in bboxes], class_labels=[b[4] for b in bboxes])
                    new_img = augmented['image']
                    new_bboxes = augmented['bboxes']
                    new_classes = augmented['class_labels']
                    
                    if not new_bboxes: continue
                    
                    new_name = f"aug_{cls_id}_{i}_{name_no_ext}"
                    cv2.imwrite(os.path.join(aug_img_dir, f"{new_name}.jpg"), new_img)
                    with open(os.path.join(aug_lbl_dir, f"{new_name}.txt"), 'w') as f:
                        for b, c in zip(new_bboxes, new_classes):
                            f.write(f"{c} {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}\n")
                except:
                    continue

    print("\n✅ PHASE 3 COMPLETE. Augmentation Summary Report printed above.")

import shutil
if __name__ == "__main__":
    run_augmentation(r"c:\Users\palan\OneDrive\Desktop\Projects\Underwater Trash Detection Project")
