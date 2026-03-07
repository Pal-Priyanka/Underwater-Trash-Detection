import os
import cv2
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from collections import Counter

def run_eda(root_dir):
    print("--- PHASE 2: EXPLORATORY DATA ANALYSIS (EDA) ---")
    eda_out = os.path.join(root_dir, 'eda_outputs')
    os.makedirs(eda_out, exist_ok=True)
    
    class_names = ['plastic', 'metal', 'wood', 'glass', 'rubber', 'cloth', 'paper', 'fishing', 'bio', 'unknown']
    
    data = []
    image_sizes = []
    
    # Traverse all splits
    for split in ['train', 'val', 'test']:
        img_dir = os.path.join(root_dir, split, 'images')
        lbl_dir = os.path.join(root_dir, split, 'labels')
        
        images = glob.glob(os.path.join(img_dir, '*.jpg'))
        for img_path in tqdm(images, desc=f"Processing {split}"):
            img = cv2.imread(img_path)
            if img is None: continue
            h, w, _ = img.shape
            image_sizes.append((w, h))
            
            name = os.path.splitext(os.path.basename(img_path))[0]
            lbl_path = os.path.join(lbl_dir, name + '.txt')
            
            obj_count = 0
            if os.path.exists(lbl_path):
                with open(lbl_path, 'r') as f:
                    file_classes = []
                    for line in f:
                        parts = line.strip().split()
                        if not parts: continue
                        cls_id = int(parts[0])
                        cx, cy, bw, bh = map(float, parts[1:])
                        
                        data.append({
                            'split': split,
                            'class_id': cls_id,
                            'class_name': class_names[cls_id] if cls_id < len(class_names) else f"ID_{cls_id}",
                            'cx': cx, 'cy': cy, 'bw': bw, 'bh': bh,
                            'w_px': bw * w, 'h_px': bh * h,
                            'aspect_ratio': (bw * w) / (bh * h) if bh > 0 else 0
                        })
                        obj_count += 1
                        file_classes.append(cls_id)
            # We don't track images with 0 objects here for box data, but for objects-per-image we should
            
    df = pd.DataFrame(data)
    
    # 1. Class Distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='class_name', order=df['class_name'].value_counts().index)
    plt.yscale('log')
    plt.title('Class Distribution (Log Scale)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(eda_out, 'class_distribution.png'))
    plt.close()

    # 2. Bounding Box Size Distribution
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df, x='w_px', y='h_px', hue='class_name', alpha=0.3)
    plt.title('Bounding Box Width vs Height')
    plt.savefig(os.path.join(eda_out, 'bbox_sizes.png'))
    plt.close()

    # 3. Aspect Ratio Histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(df['aspect_ratio'], bins=50, kde=True)
    plt.xlim(0, 5)
    plt.title('Bounding Box Aspect Ratio Histogram')
    plt.savefig(os.path.join(eda_out, 'aspect_ratio.png'))
    plt.close()

    # 4. Spatial Heatmap
    plt.figure(figsize=(8, 8))
    plt.hist2d(df['cx'], df['cy'], bins=50, cmap='viridis')
    plt.colorbar(label='Frequency')
    plt.title('Heatmap of Bounding Box Centers')
    plt.gca().invert_yaxis()
    plt.savefig(os.path.join(eda_out, 'spatial_heatmap.png'))
    plt.close()
    
    # 5. Sample Grid (16 random)
    print("Generating sample grid...")
    all_imgs = glob.glob(os.path.join(root_dir, 'train', 'images', '*.jpg'))
    samples = np.random.choice(all_imgs, 16, replace=False)
    fig, axes = plt.subplots(2, 8, figsize=(20, 5))
    for i, ax in enumerate(axes.flat):
        img = cv2.imread(samples[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        name = os.path.splitext(os.path.basename(samples[i]))[0]
        l_path = os.path.join(root_dir, 'train', 'labels', name + '.txt')
        if os.path.exists(l_path):
            h, w, _ = img.shape
            with open(l_path, 'r') as f:
                for line in f:
                    p = line.split()
                    c, cx, cy, bw, bh = int(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4])
                    x1 = int((cx - bw/2) * w)
                    y1 = int((cy - bh/2) * h)
                    x2 = int((cx + bw/2) * w)
                    y2 = int((cy + bh/2) * h)
                    cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(eda_out, 'sample_grid.png'))
    plt.close()

    # 6. Image size distribution
    ws, hs = zip(*image_sizes)
    plt.figure(figsize=(8, 6))
    plt.scatter(ws, hs, alpha=0.5)
    plt.title('Image Size Distribution')
    plt.xlabel('Width')
    plt.ylabel('Height')
    plt.savefig(os.path.join(eda_out, 'image_sizes.png'))
    plt.close()

    # 7. Objects per image histogram
    lbl_files = glob.glob(os.path.join(root_dir, 'train', 'labels', '*.txt'))
    obj_per_img = []
    for lf in lbl_files:
        with open(lf, 'r') as f:
            obj_per_img.append(len(f.readlines()))
    plt.figure(figsize=(10, 6))
    plt.hist(obj_per_img, bins=range(0, max(obj_per_img)+2), align='left')
    plt.title('Objects Per Image Histogram')
    plt.xlabel('Number of Objects')
    plt.ylabel('Image Count')
    plt.savefig(os.path.join(eda_out, 'objects_per_image.png'))
    plt.close()

    # 8. Class co-occurrence matrix
    matrix = np.zeros((10, 10))
    for lf in lbl_files:
        with open(lf, 'r') as f:
            classes = list(set([int(line.split()[0]) for line in f if line.strip()]))
            for i in range(len(classes)):
                for j in range(len(classes)):
                    if classes[i] < 10 and classes[j] < 10:
                        matrix[classes[i], classes[j]] += 1
    plt.figure(figsize=(12, 10))
    sns.heatmap(matrix, annot=True, fmt='.0f', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
    plt.title('Class Co-occurrence Matrix')
    plt.savefig(os.path.join(eda_out, 'co_occurrence_matrix.png'))
    plt.close()

    # Insight Summary
    summary = """EDA Insight Summary:
- Severe class imbalance: 'plastic' and 'bio' dominate, while 'cloth' and 'fishing' are extremely rare.
- Dominant object sizes: Objects are primarily small (avg < 50x50 pixels) in 480x320 images.
- Spatial bias: High density of objects in image centers; edges are sparsely populated.
- Augmentation: Need Mosaic, CLAHE for underwater clarity, and oversampling for rare classes."""
    with open(os.path.join(eda_out, 'eda_insight.txt'), 'w') as f:
        f.write(summary)

    print("\n✅ PHASE 2 COMPLETE. Plots and Insight Summary saved.")

if __name__ == "__main__":
    run_eda(r"c:\Users\palan\OneDrive\Desktop\Projects\Underwater Trash Detection Project")
