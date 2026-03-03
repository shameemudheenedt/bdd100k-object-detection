#!/usr/bin/env python3
"""
Fix YOLOv5 for PyTorch 2.6 compatibility
This script applies all necessary patches to make YOLOv5 work with PyTorch 2.6
"""

import os
import re

def fix_utils_file():
    """Fix the utils.py file with all necessary patches"""
    utils_path = '/home/hp/Documents/Impliment/yolov5s_bdd100k/utils/utils.py'
    
    print(f"Reading {utils_path}...")
    with open(utils_path, 'r') as f:
        content = f.read()
    
    # Fix 1: Add weights_only=False to torch.load in strip_optimizer
    print("Applying Fix 1: strip_optimizer function...")
    content = re.sub(
        r"def strip_optimizer\(f='weights/best\.pt'\):.*?\n.*?x = torch\.load\(f, map_location=torch\.device\('cpu'\)\)",
        "def strip_optimizer(f='weights/best.pt'):  # from utils.utils import *; strip_optimizer()\n    # Strip optimizer from *.pt files for lighter files (reduced by 1/2 size)\n    x = torch.load(f, map_location=torch.device('cpu'), weights_only=False)",
        content,
        flags=re.DOTALL
    )
    
    # Fix 2: Add weights_only=False to torch.load in create_pretrained
    print("Applying Fix 2: create_pretrained function...")
    content = re.sub(
        r"def create_pretrained\(f='weights/best\.pt', s='weights/pretrained\.pt'\):.*?\n.*?x = torch\.load\(f, map_location=torch\.device\('cpu'\)\)",
        "def create_pretrained(f='weights/best.pt', s='weights/pretrained.pt'):  # from utils.utils import *; create_pretrained()\n    # create pretrained checkpoint 's' from 'f' (create_pretrained(x, x) for x in glob.glob('./*.pt'))\n    x = torch.load(f, map_location=torch.device('cpu'), weights_only=False)",
        content,
        flags=re.DOTALL
    )
    
    # Fix 3: Replace interp with np.interp in ap_per_class
    print("Applying Fix 3: np.interp in ap_per_class...")
    content = content.replace(
        "r[ci] = interp(-pr_score, -conf[i], recall[:, 0])",
        "r[ci] = np.interp(-pr_score, -conf[i], recall[:, 0])"
    )
    content = content.replace(
        "p[ci] = interp(-pr_score, -conf[i], precision[:, 0])",
        "p[ci] = np.interp(-pr_score, -conf[i], precision[:, 0])"
    )
    
    # Fix 4: Replace interp with np.interp in compute_ap
    print("Applying Fix 4: np.interp in compute_ap...")
    content = content.replace(
        "ap = np.trapz(interp(x, mrec, mpre), x)",
        "ap = np.trapz(np.interp(x, mrec, mpre), x)"
    )
    
    # Fix 5: Fix output_to_target function for CUDA tensor conversion
    print("Applying Fix 5: output_to_target CUDA tensor fix...")
    old_output_to_target = r'''def output_to_target\(output, width, height\):
    """
    Convert a YOLO model output to target format
    \[batch_id, class_id, x, y, w, h, conf\]
    """
    if isinstance\(output, torch\.Tensor\):
        output = output\.cpu\(\)\.numpy\(\)

    targets = \[\]
    for i, o in enumerate\(output\):
        if o is not None:
            for pred in o:
                box = pred\[:4\].*?
                w = \(box\[2\] - box\[0\]\) / width
                h = \(box\[3\] - box\[1\]\) / height
                x = box\[0\] / width \+ w / 2
                y = box\[1\] / height \+ h / 2
                conf = pred\[4\]
                cls = int\(pred\[5\]\)

                targets\.append\(\[i, cls, x, y, w, h, conf\]\)

    return np\.array\(targets\)'''
    
    new_output_to_target = '''def output_to_target(output, width, height):
    """
    Convert a YOLO model output to target format
    [batch_id, class_id, x, y, w, h, conf]
    """
    if isinstance(output, torch.Tensor):
        output = output.cpu().numpy()

    targets = []
    for i, o in enumerate(output):
        if o is not None:
            # Convert to CPU if it's a tensor
            if isinstance(o, torch.Tensor):
                o = o.cpu().numpy()
            
            for pred in o:
                # Ensure all values are numpy
                if isinstance(pred, torch.Tensor):
                    pred = pred.cpu().numpy()
                    
                box = pred[:4]
                w = (box[2] - box[0]) / width
                h = (box[3] - box[1]) / height
                x = box[0] / width + w / 2
                y = box[1] / height + h / 2
                conf = float(pred[4])
                cls = int(pred[5])

                targets.append([i, cls, float(x), float(y), float(w), float(h), conf])

    return np.array(targets)'''
    
    content = re.sub(old_output_to_target, new_output_to_target, content, flags=re.DOTALL)
    
    # Write the fixed content back
    print(f"Writing fixed content to {utils_path}...")
    with open(utils_path, 'w') as f:
        f.write(content)
    
    print(" All fixes applied successfully!")

def fix_yaml_config():
    """Fix the bdd100k.yaml configuration file"""
    yaml_path = '/home/hp/Documents/Impliment/bdd100k_yolo/bdd100k.yaml'
    
    if not os.path.exists(yaml_path):
        print(f"  {yaml_path} not found. Skipping YAML fix.")
        return
    
    print(f"Fixing {yaml_path}...")
    
    yaml_content = """names:
- person
- rider
- car
- bus
- truck
- bike
- motor
- traffic light
- traffic sign
- train
nc: 10
train: /home/hp/Documents/Impliment/bdd100k_yolo/images/train
val: /home/hp/Documents/Impliment/bdd100k_yolo/images/val
"""
    
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(" YAML configuration fixed!")

def main():
    print("=" * 60)
    print("YOLOv5 PyTorch 2.6 Compatibility Fix Script")
    print("=" * 60)
    print()
    
    # Apply all fixes
    fix_utils_file()
    print()
    fix_yaml_config()
    
    print()
    print("=" * 60)
    print(" All fixes completed successfully!")
    print("=" * 60)
    print()
    print("You can now run:")
    print("  python model_pipeline.py --task train --epochs 3")
    print()

if __name__ == "__main__":
    main()
