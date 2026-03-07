import os
import xml.etree.ElementTree as ET

# Define class mapping and ignore list
CLASS_MAP = {
    'plastic': 0,
    'platstic': 0,
    'metal': 1,
    'wood': 2,
    'glass': 3,
    'rubber': 4,
    'cloth': 5,
    'papper': 6,
    'paper': 6,
    'fishing': 7,
    'bio': 8,
    'unknown': 9
}
IGNORE_CLASSES = ['timestamp', 'rov']

def convert_bbox(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

def preprocess_split(split_path):
    print(f"Preprocessing split: {split_path}")
    if not os.path.exists(split_path):
        print(f"Path {split_path} does not exist.")
        return

    xml_files = [f for f in os.listdir(split_path) if f.lower().endswith('.xml')]
    
    count = 0
    for xml_file in xml_files:
        xml_path = os.path.join(split_path, xml_file)
        txt_path = os.path.join(split_path, os.path.splitext(xml_file)[0] + '.txt')
        
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            size = root.find('size')
            w = int(size.find('width').text)
            h = int(size.find('height').text)
            
            if w == 0 or h == 0:
                print(f"Warning: Zero dimension in {xml_file}")
                continue

            yolo_labels = []
            for obj in root.findall('object'):
                cls_name = obj.find('name').text.lower().strip()
                
                if cls_name in IGNORE_CLASSES:
                    continue
                
                if cls_name not in CLASS_MAP:
                    print(f"Unknown class '{cls_name}' in {xml_file}, mapping to 'unknown'")
                    cls_id = CLASS_MAP['unknown']
                else:
                    cls_id = CLASS_MAP[cls_name]
                
                xmlbox = obj.find('bndbox')
                b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), 
                     float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
                bb = convert_bbox((w, h), b)
                yolo_labels.append(f"{cls_id} {' '.join([f'{a:.6f}' for a in bb])}")
            
            # Overwrite the existing .txt file with cleaned labels
            with open(txt_path, 'w') as f:
                f.write('\n'.join(yolo_labels) + '\n')
            count += 1
            
        except Exception as e:
            print(f"Error processing {xml_file}: {e}")
            
    print(f"Processed {count} files in {split_path}")

if __name__ == "__main__":
    root_dir = r"c:\Users\palan\OneDrive\Desktop\Projects\Underwater Trash Detection Project"
    for split in ['train', 'val', 'test']:
        preprocess_split(os.path.join(root_dir, split))
