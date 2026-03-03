"""
Model Evaluation and Visualization for BDD100K Object Detection
"""
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
import pandas as pd
from tqdm import tqdm


class DetectionVisualizer:
    """Visualize ground truth and predictions."""
    
    CLASSES = ['person', 'rider', 'car', 'bus', 'truck', 'bike', 'motor',
               'traffic light', 'traffic sign', 'train']
    COLORS = plt.cm.tab10(np.linspace(0, 1, 10))
    
    def __init__(self, images_dir, labels_json, output_dir='visualizations'):
        self.images_dir = Path(images_dir)
        self.labels_json = labels_json
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        with open(labels_json) as f:
            self.labels_data = {item['name']: item for item in json.load(f)}
    
    def visualize_sample(self, image_name, predictions=None):
        """Visualize single image with GT and predictions."""
        img_path = self.images_dir / image_name
        img = cv2.imread(str(img_path))
        if img is None:
            return
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        
        fig, axes = plt.subplots(1, 2 if predictions else 1, figsize=(15, 7))
        if not predictions:
            axes = [axes]
        
        # Ground truth
        gt_img = img.copy()
        if image_name in self.labels_data:
            item = self.labels_data[image_name]
            if 'labels' in item:
                for label in item['labels']:
                    if 'box2d' not in label:
                        continue
                    
                    category = label['category']
                    if category not in self.CLASSES:
                        continue
                    
                    class_id = self.CLASSES.index(category)
                    box = label['box2d']
                    
                    x1, y1 = int(box['x1']), int(box['y1'])
                    x2, y2 = int(box['x2']), int(box['y2'])
                    
                    color = tuple(int(c * 255) for c in self.COLORS[class_id][:3])
                    cv2.rectangle(gt_img, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(gt_img, category, (x1, y1-5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        axes[0].imshow(gt_img)
        axes[0].set_title('Ground Truth')
        axes[0].axis('off')
        
        # Predictions
        if predictions:
            pred_img = img.copy()
            for pred in predictions:
                class_id = int(pred['class'])
                conf = pred['confidence']
                box = pred['box']
                
                x1, y1, x2, y2 = map(int, box)
                color = tuple(int(c * 255) for c in self.COLORS[class_id][:3])
                
                cv2.rectangle(pred_img, (x1, y1), (x2, y2), color, 2)
                label = f"{self.CLASSES[class_id]} {conf:.2f}"
                cv2.putText(pred_img, label, (x1, y1-5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            axes[1].imshow(pred_img)
            axes[1].set_title('Predictions')
            axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"vis_{image_name}", dpi=150, bbox_inches='tight')
        plt.close()
    
    def visualize_failures(self, failure_cases, max_samples=20):
        """Visualize model failure cases."""
        for i, case in enumerate(failure_cases[:max_samples]):
            self.visualize_sample(case['image'], case.get('predictions'))


class PerformanceAnalyzer:
    """Analyze model performance metrics."""
    
    CLASSES = ['person', 'rider', 'car', 'bus', 'truck', 'bike', 'motor',
               'traffic light', 'traffic sign', 'train']
    
    def __init__(self, gt_json, pred_json=None):
        self.gt_json = gt_json
        self.pred_json = pred_json
        
        with open(gt_json) as f:
            self.gt_data = json.load(f)
    
    def compute_metrics(self, predictions):
        """Compute precision, recall, mAP per class."""
        metrics = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0, 'ious': []})
        
        for img_name, preds in predictions.items():
            gt_item = next((item for item in self.gt_data if item['name'] == img_name), None)
            if not gt_item or 'labels' not in gt_item:
                continue
            
            gt_boxes = []
            for label in gt_item['labels']:
                if 'box2d' in label and label['category'] in self.CLASSES:
                    gt_boxes.append({
                        'class': self.CLASSES.index(label['category']),
                        'box': [label['box2d']['x1'], label['box2d']['y1'],
                               label['box2d']['x2'], label['box2d']['y2']]
                    })
            
            matched_gt = set()
            for pred in preds:
                best_iou = 0
                best_gt_idx = -1
                
                for gt_idx, gt in enumerate(gt_boxes):
                    if gt['class'] != pred['class'] or gt_idx in matched_gt:
                        continue
                    
                    iou = self._compute_iou(pred['box'], gt['box'])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                
                class_name = self.CLASSES[pred['class']]
                if best_iou >= 0.5:
                    metrics[class_name]['tp'] += 1
                    metrics[class_name]['ious'].append(best_iou)
                    matched_gt.add(best_gt_idx)
                else:
                    metrics[class_name]['fp'] += 1
            
            for gt_idx, gt in enumerate(gt_boxes):
                if gt_idx not in matched_gt:
                    class_name = self.CLASSES[gt['class']]
                    metrics[class_name]['fn'] += 1
        
        results = {}
        for cls in self.CLASSES:
            tp = metrics[cls]['tp']
            fp = metrics[cls]['fp']
            fn = metrics[cls]['fn']
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            results[cls] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'avg_iou': np.mean(metrics[cls]['ious']) if metrics[cls]['ious'] else 0
            }
        
        return results
    
    def _compute_iou(self, box1, box2):
        """Compute IoU between two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def analyze_by_attributes(self, predictions):
        """Analyze performance by weather, scene, time."""
        attr_metrics = defaultdict(lambda: defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0}))
        
        for item in self.gt_data:
            img_name = item['name']
            if img_name not in predictions:
                continue
            
            attrs = item.get('attributes', {})
            weather = attrs.get('weather', 'unknown')
            scene = attrs.get('scene', 'unknown')
            timeofday = attrs.get('timeofday', 'unknown')
            
            # Simplified metric computation
            gt_count = sum(1 for l in item.get('labels', []) if 'box2d' in l)
            pred_count = len(predictions[img_name])
            
            for attr_type, attr_val in [('weather', weather), ('scene', scene), ('timeofday', timeofday)]:
                attr_metrics[attr_type][attr_val]['tp'] += min(gt_count, pred_count)
                attr_metrics[attr_type][attr_val]['fp'] += max(0, pred_count - gt_count)
                attr_metrics[attr_type][attr_val]['fn'] += max(0, gt_count - pred_count)
        
        return dict(attr_metrics)
    
    def identify_failure_cases(self, predictions, threshold=0.3):
        """Identify images where model performs poorly."""
        failures = []
        
        for item in self.gt_data:
            img_name = item['name']
            if img_name not in predictions:
                continue
            
            gt_count = sum(1 for l in item.get('labels', []) if 'box2d' in l)
            pred_count = len(predictions[img_name])
            
            if gt_count == 0:
                continue
            
            recall_approx = min(pred_count, gt_count) / gt_count
            
            if recall_approx < threshold:
                failures.append({
                    'image': img_name,
                    'gt_count': gt_count,
                    'pred_count': pred_count,
                    'recall_approx': recall_approx,
                    'attributes': item.get('attributes', {}),
                    'predictions': predictions[img_name]
                })
        
        return sorted(failures, key=lambda x: x['recall_approx'])


def plot_metrics(metrics, output_dir='evaluation_output'):
    """Plot performance metrics."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    df = pd.DataFrame(metrics).T
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    df['precision'].plot(kind='bar', ax=axes[0, 0], color='skyblue')
    axes[0, 0].set_title('Precision by Class')
    axes[0, 0].set_ylabel('Precision')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].set_ylim([0, 1])
    
    df['recall'].plot(kind='bar', ax=axes[0, 1], color='lightcoral')
    axes[0, 1].set_title('Recall by Class')
    axes[0, 1].set_ylabel('Recall')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].set_ylim([0, 1])
    
    df['f1'].plot(kind='bar', ax=axes[1, 0], color='lightgreen')
    axes[1, 0].set_title('F1 Score by Class')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].set_ylim([0, 1])
    
    df['avg_iou'].plot(kind='bar', ax=axes[1, 1], color='plum')
    axes[1, 1].set_title('Average IoU by Class')
    axes[1, 1].set_ylabel('IoU')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_by_class.png', dpi=150)
    plt.close()
    
    # Confusion-like visualization
    fig, ax = plt.subplots(figsize=(10, 8))
    confusion_data = df[['tp', 'fp', 'fn']].values
    sns.heatmap(confusion_data, annot=True, fmt='g', cmap='YlOrRd',
                xticklabels=['TP', 'FP', 'FN'], yticklabels=df.index, ax=ax)
    ax.set_title('Detection Statistics by Class')
    plt.tight_layout()
    plt.savefig(output_dir / 'detection_stats.png', dpi=150)
    plt.close()


if __name__ == '__main__':
    gt_json = '/home/hp/Documents/Impliment/data/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_val.json'
    images_dir = '/home/hp/Documents/Impliment/data/bdd100k_images_100k/bdd100k/images/100k/val'
    
    print("Testing evaluation framework...")
    
    # Test visualizer with ground truth only
    visualizer = DetectionVisualizer(images_dir, gt_json)
    
    with open(gt_json) as f:
        data = json.load(f)
    
    # Visualize first 5 samples with ground truth
    print(f"Generating visualizations for {min(5, len(data))} samples...")
    for i, item in enumerate(data[:5]):
        print(f"  Processing {item['name']}...")
        visualizer.visualize_sample(item['name'])
    
    print(f"\nVisualizations saved to: {visualizer.output_dir}")
    print("Evaluation framework ready. Add predictions for full metrics.")
