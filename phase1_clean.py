import os
import shutil
from PIL import Image

def clean_dataset(root_dir):
    print("--- PHASE 1: DATA CLEANING & PREPROCESSING ---")
    
    splits = ['train', 'val', 'test']
    total_fixed_boxes = 0
    total_converted_imgs = 0
    orphans_removed = 0
    unlabeled_moved = 0
    
    unlabeled_dir = os.path.join(root_dir, 'unlabeled')
    os.makedirs(unlabeled_dir, exist_ok=True)
    
    for split in splits:
        split_path = os.path.join(root_dir, split)
        if not os.path.exists(split_path): continue
        
        img_dir = os.path.join(split_path, 'images')
        lbl_dir = os.path.join(split_path, 'labels')
        
        # In case it's not restructured yet (though I did it earlier, let's be safe)
        if not os.path.exists(img_dir): img_dir = split_path
        if not os.path.exists(lbl_dir): lbl_dir = split_path
        
        images = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        labels = [f for f in os.listdir(lbl_dir) if f.lower().endswith('.txt')]
        
        img_stems = {os.path.splitext(f)[0]: f for f in images}
        lbl_stems = {os.path.splitext(f)[0]: f for f in labels}
        
        # 1. Handle missing labels (unlabeled images)
        missing = img_stems.keys() - lbl_stems.keys()
        for stem in missing:
            f_name = img_stems[stem]
            shutil.move(os.path.join(img_dir, f_name), os.path.join(unlabeled_dir, f_name))
            unlabeled_moved += 1
            
        # 2. Handle orphan labels
        orphans = lbl_stems.keys() - img_stems.keys()
        for stem in orphans:
            os.remove(os.path.join(lbl_dir, lbl_stems[stem]))
            orphans_removed += 1
            
        # 3. Standardize to .jpg and validate boxes
        curr_images = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        for img_name in curr_images:
            name_no_ext, ext = os.path.splitext(img_name)
            img_path = os.path.join(img_dir, img_name)
            
            # Convert to jpg if not already
            if ext.lower() != '.jpg':
                try:
                    with Image.open(img_path) as img:
                        rgb_img = img.convert('RGB')
                        rgb_img.save(os.path.join(img_dir, name_no_ext + '.jpg'), 'JPEG')
                    os.remove(img_path)
                    total_converted_imgs += 1
                except:
                    print(f"Failed to convert {img_path}")
            
            # Validate boxes in corresponding label
            lbl_path = os.path.join(lbl_dir, name_no_ext + '.txt')
            if os.path.exists(lbl_path):
                updated_lines = []
                fixed = False
                with open(lbl_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if not parts: continue
                        cls_id = parts[0]
                        coords = [float(x) for x in parts[1:]]
                        
                        new_coords = []
                        for c in coords:
                            if c < 0:
                                c = 0.0
                                fixed = True
                            elif c > 1:
                                c = 1.0
                                fixed = True
                            new_coords.append(c)
                        
                        updated_lines.append(f"{cls_id} " + " ".join([f"{x:.6f}" for x in new_coords]))
                
                if fixed:
                    with open(lbl_path, 'w') as f:
                        f.write("\n".join(updated_lines) + "\n")
                    total_fixed_boxes += 1

    print("\n✅ PHASE 1 COMPLETE")
    print(f"Orphan labels removed: {orphans_removed}")
    print(f"Unlabeled images moved to /unlabeled: {unlabeled_moved}")
    print(f"Images converted to .jpg: {total_converted_imgs}")
    print(f"Label files with clamped coordinates: {total_fixed_boxes}")

if __name__ == "__main__":
    clean_dataset(r"c:\Users\palan\OneDrive\Desktop\Projects\Underwater Trash Detection Project")
