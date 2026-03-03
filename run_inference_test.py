#!/usr/bin/env python3
"""
Run inference on BDD100K test set
"""
import sys
sys.path.insert(0, '/home/hp/Documents/Impliment/yolov5s_bdd100k')

from pathlib import Path
import torch
import cv2
import json
from models.experimental import attempt_load
from utils.utils import non_max_suppression, scale_coords
from utils.torch_utils import select_device
import numpy as np

# Configuration
WEIGHTS = '/home/hp/Documents/Impliment/yolov5s_bdd100k/runs/exp1_yolov5s_bdd/weights/best_yolov5s_bdd.pt'
TEST_DIR = '/home/hp/Documents/Impliment/data/bdd100k_images_100k/bdd100k/images/100k/test'
OUTPUT_JSON = '/home/hp/Documents/Impliment/test_predictions.json'
CLASSES = ['person', 'rider', 'car', 'bus', 'truck', 'bike', 'motor',
           'traffic light', 'traffic sign', 'train']

def preprocess(img, img_size=640, device='cpu'):
    img = cv2.resize(img, (img_size, img_size))
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.float() / 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    return img

if __name__ == '__main__':
    print("Initializing detector...")
    device = select_device('')
    model = attempt_load(WEIGHTS, map_location=device)
    model.eval()
    
    print(f"Running inference on test set: {TEST_DIR}")
    test_dir = Path(TEST_DIR)
    results = {}
    
    img_files = list(test_dir.glob('*.jpg'))[:100]
    print(f"Found {len(img_files)} images to process")
    
    for idx, img_path in enumerate(img_files):
        img0 = cv2.imread(str(img_path))
        if img0 is None:
            continue
        
        img = preprocess(img0, device=device)
        
        with torch.no_grad():
            pred = model(img)[0]
        
        pred = non_max_suppression(pred, 0.25, 0.45)
        
        detections = []
        for det in pred:
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
                
                for *xyxy, conf, cls in det:
                    cls_idx = int(cls)
                    if cls_idx < len(CLASSES):
                        detections.append({
                            'class': cls_idx,
                            'class_name': CLASSES[cls_idx],
                            'confidence': float(conf),
                            'box': [int(x) for x in xyxy]
                        })
        
        results[img_path.name] = detections
        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{len(img_files)} images")
    
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nInference complete!")
    print(f"Total images processed: {len(results)}")
    print(f"Results saved to: {OUTPUT_JSON}")
