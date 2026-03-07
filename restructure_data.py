import os
import shutil

def restructure_split(split_path):
    print(f"Restructuring split: {split_path}")
    if not os.path.exists(split_path):
        print(f"Path {split_path} does not exist.")
        return

    images_dir = os.path.join(split_path, 'images')
    labels_dir = os.path.join(split_path, 'labels')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    files = os.listdir(split_path)
    img_count = 0
    lbl_count = 0
    
    for f in files:
        f_path = os.path.join(split_path, f)
        if os.path.isdir(f_path):
            continue
            
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            shutil.move(f_path, os.path.join(images_dir, f))
            img_count += 1
        elif f.lower().endswith('.txt'):
            shutil.move(f_path, os.path.join(labels_dir, f))
            lbl_count += 1
            
    print(f"Moved {img_count} images and {lbl_count} labels in {split_path}")

if __name__ == "__main__":
    root_dir = r"c:\Users\palan\OneDrive\Desktop\Projects\Underwater Trash Detection Project"
    for split in ['train', 'val', 'test']:
        restructure_split(os.path.join(root_dir, split))
