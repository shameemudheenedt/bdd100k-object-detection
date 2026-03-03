"""
Inference script for BDD100K object detection
"""
import torch
import cv2
import numpy as np
from pathlib import Path
import json
import sys
import random

sys.path.insert(0, '/home/hp/Documents/Impliment/yolov5s_bdd100k')

from models.experimental import attempt_load
from utils.utils import non_max_suppression, scale_coords
from utils.torch_utils import select_device


class BDD100KDetector:
    """YOLOv5 detector for BDD100K."""
    
    CLASSES = ['person', 'rider', 'car', 'bus', 'truck', 'bike', 'motor',
               'traffic light', 'traffic sign', 'train']
    COLORS = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(CLASSES))]
    
    def __init__(self, weights_path, img_size=640, conf_thres=0.25, iou_thres=0.45):
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        
        self.device = select_device('')
        self.model = attempt_load(weights_path, map_location=self.device)
        self.model.eval()
        
    def preprocess(self, img):
        """Preprocess image for inference."""
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.float() / 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return img
    
    def detect(self, img_path, save_img=False, output_dir='inference_output'):
        """Run detection on single image."""
        img0 = cv2.imread(str(img_path))
        if img0 is None:
            return [], None
        
        img = self.preprocess(img0)
        
        with torch.no_grad():
            pred = self.model(img)[0]
        
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)
        
        detections = []
        img_with_boxes = img0.copy()
        
        for det in pred:
            if det is not None and len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
                
                for *xyxy, conf, cls in det:
                    cls_idx = int(cls)
                    if cls_idx < len(self.CLASSES):
                        detections.append({
                            'class': cls_idx,
                            'class_name': self.CLASSES[cls_idx],
                            'confidence': float(conf),
                            'box': [int(x) for x in xyxy]
                        })
                        
                        if save_img:
                            x1, y1, x2, y2 = [int(x) for x in xyxy]
                            color = self.COLORS[cls_idx]
                            cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, 2)
                            label = f'{self.CLASSES[cls_idx]} {conf:.2f}'
                            cv2.putText(img_with_boxes, label, (x1, y1-10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        if save_img:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
            save_path = output_path / Path(img_path).name
            cv2.imwrite(str(save_path), img_with_boxes)
            return detections, str(save_path)
        
        return detections, None
    
    def detect_batch(self, img_dir, output_json='predictions.json', save_img=False, output_dir='inference_output'):
        """Run detection on directory of images."""
        img_dir = Path(img_dir)
        results = {}
        
        for img_path in img_dir.glob('*.jpg'):
            detections, img_path_saved = self.detect(img_path, save_img, output_dir)
            results[img_path.name] = detections
            if save_img:
                print(f"Processed {img_path.name}: {len(detections)} detections, saved to {img_path_saved}")
            else:
                print(f"Processed {img_path.name}: {len(detections)} detections")
        
        with open(output_json, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Saved predictions to {output_json}")
        return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', required=True, help='Model weights path')
    parser.add_argument('--source', required=True, help='Image or directory path')
    parser.add_argument('--output', default='predictions.json', help='Output JSON')
    parser.add_argument('--output-dir', default='inference_output', help='Output directory for images')
    parser.add_argument('--save-img', action='store_true', help='Save images with bounding boxes')
    parser.add_argument('--img-size', type=int, default=640)
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--iou', type=float, default=0.45)
    args = parser.parse_args()
    
    detector = BDD100KDetector(args.weights, args.img_size, args.conf, args.iou)
    
    source_path = Path(args.source)
    if source_path.is_file():
        detections, img_saved = detector.detect(source_path, args.save_img, args.output_dir)
        print(f"Detections: {len(detections)}")
        if img_saved:
            print(f"Image saved to: {img_saved}")
    else:
        detector.detect_batch(source_path, args.output, args.save_img, args.output_dir)
