"""
YOLOv5 Model Training and Evaluation for BDD100K
"""
import torch
import yaml
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import cv2
from tqdm import tqdm


class BDD100KConverter:
    """Convert BDD100K format to YOLO format."""
    
    CLASSES = ['person', 'rider', 'car', 'bus', 'truck', 'bike', 'motor', 
               'traffic light', 'traffic sign', 'train']
    
    def __init__(self, labels_path, images_path, output_path):
        self.labels_path = Path(labels_path)
        self.images_path = Path(images_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
    def convert(self):
        """Convert BDD100K labels to YOLO format."""
        with open(self.labels_path) as f:
            data = json.load(f)
        
        labels_dir = self.output_path / 'labels'
        labels_dir.mkdir(exist_ok=True)
        
        for item in tqdm(data, desc=f"Converting {self.labels_path.name}"):
            img_name = item['name']
            img_path = self.images_path / img_name
            
            if not img_path.exists():
                continue
            
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            h, w = img.shape[:2]
            
            label_file = labels_dir / f"{img_path.stem}.txt"
            
            with open(label_file, 'w') as f:
                if 'labels' not in item:
                    continue
                    
                for label in item['labels']:
                    if 'box2d' not in label:
                        continue
                    
                    category = label['category']
                    if category not in self.CLASSES:
                        continue
                    
                    class_id = self.CLASSES.index(category)
                    box = label['box2d']
                    
                    x_center = ((box['x1'] + box['x2']) / 2) / w
                    y_center = ((box['y1'] + box['y2']) / 2) / h
                    width = (box['x2'] - box['x1']) / w
                    height = (box['y2'] - box['y1']) / h
                    
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")


def prepare_dataset():
    """Prepare BDD100K dataset in YOLO format."""
    base_dir = Path('/home/hp/Documents/Impliment/data')
    output_dir = Path('/home/hp/Documents/Impliment/bdd100k_yolo')
    
    # Convert train
    train_converter = BDD100KConverter(
        base_dir / 'bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_train.json',
        base_dir / 'bdd100k_images_100k/bdd100k/images/100k/train',
        output_dir / 'train'
    )
    train_converter.convert()
    
    # Convert val
    val_converter = BDD100KConverter(
        base_dir / 'bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_val.json',
        base_dir / 'bdd100k_images_100k/bdd100k/images/100k/val',
        output_dir / 'val'
    )
    val_converter.convert()
    
    # Create dataset YAML
    yaml_content = {
        'path': str(output_dir),
        'train': 'train',
        'val': 'val',
        'nc': 10,
        'names': BDD100KConverter.CLASSES
    }
    
    with open(output_dir / 'bdd100k.yaml', 'w') as f:
        yaml.dump(yaml_content, f)
    
    print(f"Dataset prepared at {output_dir}")


class YOLOv5Trainer:
    """YOLOv5 training wrapper."""
    
    def __init__(self, yolo_dir, data_yaml, weights='yolov5s.pt'):
        self.yolo_dir = Path(yolo_dir)
        self.data_yaml = data_yaml
        self.weights = weights
        
    def train_one_epoch(self, img_size=640, batch_size=16, epochs=1):
        """Train for one epoch on subset."""
        import subprocess
        
        weights_path = self.yolo_dir / 'weights' / self.weights
        cfg_path = self.yolo_dir / 'models/custom_yolov5s.yaml'
        
        if not cfg_path.exists():
            cfg_path = self.yolo_dir / 'models/yolov5s.yaml'
        
        device = '0' if torch.cuda.is_available() else 'cpu'
        
        cmd = [
            'python', str(self.yolo_dir / 'train.py'),
            '--data', self.data_yaml,
            '--cfg', str(cfg_path),
            '--weights', str(weights_path),
            '--epochs', str(epochs),
            '--batch-size', str(batch_size),
            '--img-size', str(img_size),
            '--device', device,
            '--name', 'bdd100k_exp'
        ]
        
        print(f"Training on device: {device}")
        print(f"Command: {' '.join(cmd)}")
        print(f"\nNote: Training 1 epoch on 70K images takes 30-45 minutes.")
        
        result = subprocess.run(cmd, cwd=str(self.yolo_dir.parent))
        return result.returncode == 0


class ModelEvaluator:
    """Evaluate YOLOv5 model on BDD100K."""
    
    def __init__(self, model_path, data_yaml, yolo_dir):
        self.model_path = model_path
        self.data_yaml = data_yaml
        self.yolo_dir = Path(yolo_dir)
        
    def evaluate(self):
        """Run evaluation on validation set."""
        import subprocess
        
        device = '0' if torch.cuda.is_available() else 'cpu'
        
        # Convert to absolute path
        weights_path = Path(self.model_path).resolve()
        
        cmd = [
            'python', str(self.yolo_dir / 'test.py'),
            '--data', self.data_yaml,
            '--weights', str(weights_path),
            '--batch-size', '32',
            '--img-size', '640',
            '--conf-thres', '0.001',
            '--iou-thres', '0.6',
            '--task', 'val',
            '--device', device,
            '--save-json',
            '--verbose'
        ]
        
        print(f"Evaluating on device: {device}")
        print(f"Model: {weights_path}")
        print(f"Data: {self.data_yaml}")
        print()
        
        result = subprocess.run(cmd, cwd=str(self.yolo_dir.parent))
        
        if result.returncode == 0:
            print("\nEvaluation completed successfully!")
            print("Results saved in YOLOv5 directory")
        else:
            print("\nEvaluation failed. Check logs above.")
        
        return result.returncode == 0


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=['prepare', 'train', 'eval'], required=True)
    parser.add_argument('--weights', default='yolov5s.pt')
    parser.add_argument('--epochs', type=int, default=1)
    args = parser.parse_args()
    
    if args.task == 'prepare':
        prepare_dataset()
    elif args.task == 'train':
        trainer = YOLOv5Trainer(
            '/home/hp/Documents/Impliment/yolov5s_bdd100k',
            '/home/hp/Documents/Impliment/bdd100k_yolo/bdd100k.yaml',
            args.weights
        )
        success = trainer.train_one_epoch(epochs=args.epochs)
        if success:
            print("\nTraining completed successfully!")
            print("Weights saved to: runs/train/bdd100k_exp/weights/")
        else:
            print("\nTraining failed. Check logs above.")
    elif args.task == 'eval':
        evaluator = ModelEvaluator(
            args.weights,
            '/home/hp/Documents/Impliment/bdd100k_yolo/bdd100k.yaml',
            '/home/hp/Documents/Impliment/yolov5s_bdd100k'
        )
        evaluator.evaluate()
