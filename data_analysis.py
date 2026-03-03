"""
BDD100K Dataset Analysis Module
Analyzes object detection dataset including class distribution, splits, and anomalies.
"""
import json
import os
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import pandas as pd


class BDD100KParser:
    """Parser for BDD100K dataset JSON labels."""
    
    def __init__(self, labels_path: str, images_path: str):
        """
        Initialize parser with paths to labels and images.
        
        Args:
            labels_path: Path to JSON labels file
            images_path: Path to images directory
        """
        self.labels_path = labels_path
        self.images_path = images_path
        self.data = None
        self.stats = defaultdict(lambda: defaultdict(int))
        
    def load_labels(self) -> List[Dict]:
        """Load and parse JSON labels file."""
        with open(self.labels_path, 'r') as f:
            self.data = json.load(f)
        return self.data
    
    def get_class_distribution(self) -> Dict[str, int]:
        """Calculate distribution of object detection classes."""
        class_counts = Counter()
        for item in self.data:
            if 'labels' in item:
                for label in item['labels']:
                    if 'box2d' in label:
                        class_counts[label['category']] += 1
        return dict(class_counts)
    
    def get_bbox_stats(self) -> Dict[str, Dict]:
        """Calculate bounding box statistics per class."""
        bbox_stats = defaultdict(lambda: {'widths': [], 'heights': [], 'areas': []})
        
        for item in self.data:
            if 'labels' in item:
                for label in item['labels']:
                    if 'box2d' in label:
                        box = label['box2d']
                        width = box['x2'] - box['x1']
                        height = box['y2'] - box['y1']
                        area = width * height
                        
                        category = label['category']
                        bbox_stats[category]['widths'].append(width)
                        bbox_stats[category]['heights'].append(height)
                        bbox_stats[category]['areas'].append(area)
        
        return dict(bbox_stats)
    
    def get_attribute_distribution(self) -> Dict:
        """Analyze weather, scene, and time of day distributions."""
        attributes = {
            'weather': Counter(),
            'scene': Counter(),
            'timeofday': Counter()
        }
        
        for item in self.data:
            if 'attributes' in item:
                for key in attributes.keys():
                    if key in item['attributes']:
                        attributes[key][item['attributes'][key]] += 1
        
        return {k: dict(v) for k, v in attributes.items()}
    
    def get_occlusion_truncation_stats(self) -> Dict:
        """Analyze occlusion and truncation statistics per class."""
        stats = defaultdict(lambda: {'occluded': 0, 'truncated': 0, 'total': 0})
        
        for item in self.data:
            if 'labels' in item:
                for label in item['labels']:
                    if 'box2d' in label:
                        category = label['category']
                        stats[category]['total'] += 1
                        if label.get('attributes', {}).get('occluded', False):
                            stats[category]['occluded'] += 1
                        if label.get('attributes', {}).get('truncated', False):
                            stats[category]['truncated'] += 1
        
        return dict(stats)
    
    def get_objects_per_image_stats(self) -> Dict:
        """Calculate statistics on number of objects per image."""
        objects_per_image = []
        class_per_image = defaultdict(list)
        
        for item in self.data:
            if 'labels' in item:
                total_objects = sum(1 for l in item['labels'] if 'box2d' in l)
                objects_per_image.append(total_objects)
                
                class_counts = Counter()
                for label in item['labels']:
                    if 'box2d' in label:
                        class_counts[label['category']] += 1
                
                for cls, count in class_counts.items():
                    class_per_image[cls].append(count)
        
        return {
            'overall': {
                'mean': np.mean(objects_per_image),
                'median': np.median(objects_per_image),
                'std': np.std(objects_per_image),
                'max': np.max(objects_per_image),
                'min': np.min(objects_per_image)
            },
            'per_class': {cls: {
                'mean': np.mean(counts),
                'median': np.median(counts),
                'std': np.std(counts)
            } for cls, counts in class_per_image.items()}
        }


class DatasetAnalyzer:
    """Comprehensive analyzer for BDD100K dataset."""
    
    def __init__(self, train_labels: str, val_labels: str, 
                 train_images: str, val_images: str, output_dir: str = 'analysis_output'):
        """
        Initialize analyzer with train and validation paths.
        
        Args:
            train_labels: Path to training labels JSON
            val_labels: Path to validation labels JSON
            train_images: Path to training images directory
            val_images: Path to validation images directory
            output_dir: Directory to save analysis outputs
        """
        self.train_parser = BDD100KParser(train_labels, train_images)
        self.val_parser = BDD100KParser(val_labels, val_images)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def run_full_analysis(self):
        """Execute complete dataset analysis pipeline."""
        print("Loading datasets...")
        self.train_parser.load_labels()
        self.val_parser.load_labels()
        
        print("Analyzing class distributions...")
        self.analyze_class_distribution()
        
        print("Analyzing bounding box statistics...")
        self.analyze_bbox_stats()
        
        print("Analyzing dataset attributes...")
        self.analyze_attributes()
        
        print("Analyzing occlusion and truncation...")
        self.analyze_occlusion_truncation()
        
        print("Analyzing objects per image...")
        self.analyze_objects_per_image()
        
        print("Detecting anomalies...")
        self.detect_anomalies()
        
        print("Generating summary report...")
        self.generate_summary_report()
        
        print(f"Analysis complete! Results saved to {self.output_dir}")
    
    def analyze_class_distribution(self):
        """Analyze and visualize class distribution."""
        train_dist = self.train_parser.get_class_distribution()
        val_dist = self.val_parser.get_class_distribution()
        
        df = pd.DataFrame({
            'Train': train_dist,
            'Val': val_dist
        }).fillna(0)
        
        df.to_csv(self.output_dir / 'class_distribution.csv')
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        df.plot(kind='bar', ax=axes[0])
        axes[0].set_title('Class Distribution: Train vs Val')
        axes[0].set_xlabel('Class')
        axes[0].set_ylabel('Count')
        axes[0].tick_params(axis='x', rotation=45)
        
        df['Train_Ratio'] = df['Train'] / df['Train'].sum()
        df['Val_Ratio'] = df['Val'] / df['Val'].sum()
        df[['Train_Ratio', 'Val_Ratio']].plot(kind='bar', ax=axes[1])
        axes[1].set_title('Class Distribution: Normalized')
        axes[1].set_xlabel('Class')
        axes[1].set_ylabel('Ratio')
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'class_distribution.png', dpi=150)
        plt.close()
    
    def analyze_bbox_stats(self):
        """Analyze bounding box statistics."""
        train_bbox = self.train_parser.get_bbox_stats()
        
        stats_summary = []
        for cls, data in train_bbox.items():
            stats_summary.append({
                'Class': cls,
                'Avg_Width': np.mean(data['widths']),
                'Avg_Height': np.mean(data['heights']),
                'Avg_Area': np.mean(data['areas']),
                'Std_Width': np.std(data['widths']),
                'Std_Height': np.std(data['heights']),
                'Min_Area': np.min(data['areas']),
                'Max_Area': np.max(data['areas'])
            })
        
        df = pd.DataFrame(stats_summary)
        df.to_csv(self.output_dir / 'bbox_statistics.csv', index=False)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        df.plot(x='Class', y='Avg_Width', kind='bar', ax=axes[0, 0], legend=False)
        axes[0, 0].set_title('Average Width by Class')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        df.plot(x='Class', y='Avg_Height', kind='bar', ax=axes[0, 1], legend=False)
        axes[0, 1].set_title('Average Height by Class')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        df.plot(x='Class', y='Avg_Area', kind='bar', ax=axes[1, 0], legend=False)
        axes[1, 0].set_title('Average Area by Class')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        areas = []
        labels = []
        for cls, data in train_bbox.items():
            areas.extend(data['areas'])
            labels.extend([cls] * len(data['areas']))
        
        axes[1, 1].hist([train_bbox[cls]['areas'] for cls in train_bbox.keys()], 
                       bins=50, label=list(train_bbox.keys()), alpha=0.6)
        axes[1, 1].set_title('Area Distribution by Class')
        axes[1, 1].set_xlabel('Area (pixels²)')
        axes[1, 1].set_yscale('log')
        axes[1, 1].legend(fontsize=8)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'bbox_statistics.png', dpi=150)
        plt.close()
    
    def analyze_attributes(self):
        """Analyze dataset attributes (weather, scene, time)."""
        train_attrs = self.train_parser.get_attribute_distribution()
        val_attrs = self.val_parser.get_attribute_distribution()
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for idx, attr in enumerate(['weather', 'scene', 'timeofday']):
            df = pd.DataFrame({
                'Train': train_attrs[attr],
                'Val': val_attrs[attr]
            }).fillna(0)
            
            df.to_csv(self.output_dir / f'{attr}_distribution.csv')
            
            df.plot(kind='bar', ax=axes[idx])
            axes[idx].set_title(f'{attr.capitalize()} Distribution')
            axes[idx].set_xlabel(attr.capitalize())
            axes[idx].set_ylabel('Count')
            axes[idx].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'attributes_distribution.png', dpi=150)
        plt.close()
    
    def analyze_occlusion_truncation(self):
        """Analyze occlusion and truncation statistics."""
        train_stats = self.train_parser.get_occlusion_truncation_stats()
        
        data = []
        for cls, stats in train_stats.items():
            data.append({
                'Class': cls,
                'Total': stats['total'],
                'Occluded': stats['occluded'],
                'Truncated': stats['truncated'],
                'Occluded_Ratio': stats['occluded'] / stats['total'] if stats['total'] > 0 else 0,
                'Truncated_Ratio': stats['truncated'] / stats['total'] if stats['total'] > 0 else 0
            })
        
        df = pd.DataFrame(data)
        df.to_csv(self.output_dir / 'occlusion_truncation_stats.csv', index=False)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        df.plot(x='Class', y='Occluded_Ratio', kind='bar', ax=axes[0], legend=False)
        axes[0].set_title('Occlusion Ratio by Class')
        axes[0].set_ylabel('Ratio')
        axes[0].tick_params(axis='x', rotation=45)
        
        df.plot(x='Class', y='Truncated_Ratio', kind='bar', ax=axes[1], legend=False)
        axes[1].set_title('Truncation Ratio by Class')
        axes[1].set_ylabel('Ratio')
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'occlusion_truncation.png', dpi=150)
        plt.close()
    
    def analyze_objects_per_image(self):
        """Analyze objects per image statistics."""
        train_stats = self.train_parser.get_objects_per_image_stats()
        val_stats = self.val_parser.get_objects_per_image_stats()
        
        summary = {
            'Split': ['Train', 'Val'],
            'Mean_Objects': [train_stats['overall']['mean'], val_stats['overall']['mean']],
            'Median_Objects': [train_stats['overall']['median'], val_stats['overall']['median']],
            'Std_Objects': [train_stats['overall']['std'], val_stats['overall']['std']],
            'Max_Objects': [train_stats['overall']['max'], val_stats['overall']['max']],
            'Min_Objects': [train_stats['overall']['min'], val_stats['overall']['min']]
        }
        
        df = pd.DataFrame(summary)
        df.to_csv(self.output_dir / 'objects_per_image_stats.csv', index=False)
        
        with open(self.output_dir / 'objects_per_image_detailed.txt', 'w') as f:
            f.write("TRAIN SET - Objects per Image by Class:\n")
            f.write("=" * 50 + "\n")
            for cls, stats in train_stats['per_class'].items():
                f.write(f"{cls}: mean={stats['mean']:.2f}, median={stats['median']:.2f}, std={stats['std']:.2f}\n")
            
            f.write("\n\nVAL SET - Objects per Image by Class:\n")
            f.write("=" * 50 + "\n")
            for cls, stats in val_stats['per_class'].items():
                f.write(f"{cls}: mean={stats['mean']:.2f}, median={stats['median']:.2f}, std={stats['std']:.2f}\n")
    
    def detect_anomalies(self):
        """Detect anomalies in the dataset."""
        anomalies = []
        
        train_bbox = self.train_parser.get_bbox_stats()
        
        for cls, data in train_bbox.items():
            areas = np.array(data['areas'])
            q1, q3 = np.percentile(areas, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - 3 * iqr
            upper_bound = q3 + 3 * iqr
            
            outliers = np.sum((areas < lower_bound) | (areas > upper_bound))
            
            anomalies.append({
                'Class': cls,
                'Total_Instances': len(areas),
                'Outliers': outliers,
                'Outlier_Ratio': outliers / len(areas) if len(areas) > 0 else 0,
                'Very_Small_(<100px)': np.sum(areas < 100),
                'Very_Large_(>100000px)': np.sum(areas > 100000)
            })
        
        df = pd.DataFrame(anomalies)
        df.to_csv(self.output_dir / 'anomalies.csv', index=False)
    
    def generate_summary_report(self):
        """Generate comprehensive summary report."""
        train_dist = self.train_parser.get_class_distribution()
        val_dist = self.val_parser.get_class_distribution()
        
        with open(self.output_dir / 'ANALYSIS_REPORT.md', 'w') as f:
            f.write("# BDD100K Object Detection Dataset Analysis Report\n\n")
            
            f.write("## Dataset Overview\n\n")
            f.write(f"- **Training Images**: {len(self.train_parser.data)}\n")
            f.write(f"- **Validation Images**: {len(self.val_parser.data)}\n")
            f.write(f"- **Total Training Objects**: {sum(train_dist.values())}\n")
            f.write(f"- **Total Validation Objects**: {sum(val_dist.values())}\n\n")
            
            f.write("## Class Distribution\n\n")
            f.write("| Class | Train Count | Val Count | Train % | Val % |\n")
            f.write("|-------|-------------|-----------|---------|-------|\n")
            
            total_train = sum(train_dist.values())
            total_val = sum(val_dist.values())
            
            all_classes = set(train_dist.keys()) | set(val_dist.keys())
            for cls in sorted(all_classes):
                train_count = train_dist.get(cls, 0)
                val_count = val_dist.get(cls, 0)
                train_pct = (train_count / total_train * 100) if total_train > 0 else 0
                val_pct = (val_count / total_val * 100) if total_val > 0 else 0
                f.write(f"| {cls} | {train_count} | {val_count} | {train_pct:.2f}% | {val_pct:.2f}% |\n")
            
            f.write("\n## Key Findings\n\n")
            f.write("### Class Imbalance\n")
            most_common = max(train_dist.items(), key=lambda x: x[1])
            least_common = min(train_dist.items(), key=lambda x: x[1])
            f.write(f"- Most common class: **{most_common[0]}** ({most_common[1]} instances)\n")
            f.write(f"- Least common class: **{least_common[0]}** ({least_common[1]} instances)\n")
            f.write(f"- Imbalance ratio: **{most_common[1] / least_common[1]:.2f}:1**\n\n")
            
            f.write("### Recommendations\n")
            f.write("1. Consider data augmentation for underrepresented classes\n")
            f.write("2. Use weighted loss functions to handle class imbalance\n")
            f.write("3. Monitor per-class performance during training\n")
            f.write("4. Apply techniques like focal loss for hard examples\n\n")
            
            f.write("## Generated Visualizations\n\n")
            f.write("- `class_distribution.png`: Class distribution comparison\n")
            f.write("- `bbox_statistics.png`: Bounding box statistics\n")
            f.write("- `attributes_distribution.png`: Weather, scene, and time distributions\n")
            f.write("- `occlusion_truncation.png`: Occlusion and truncation analysis\n")
            f.write("- `anomalies.csv`: Detected anomalies in the dataset\n")


if __name__ == '__main__':
    # Configure paths - use environment variable or default
    BASE_DIR = Path(os.getenv('DATA_DIR', '/app/data'))
    TRAIN_LABELS = BASE_DIR / 'bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_train.json'
    VAL_LABELS = BASE_DIR / 'bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_val.json'
    TRAIN_IMAGES = BASE_DIR / 'bdd100k_images_100k/bdd100k/images/100k/train'
    VAL_IMAGES = BASE_DIR / 'bdd100k_images_100k/bdd100k/images/100k/val'
    
    # Run analysis
    analyzer = DatasetAnalyzer(
        str(TRAIN_LABELS),
        str(VAL_LABELS),
        str(TRAIN_IMAGES),
        str(VAL_IMAGES),
        output_dir='analysis_output'
    )
    
    analyzer.run_full_analysis()
