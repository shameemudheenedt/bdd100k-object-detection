# BDD100K Object Detection Dataset Analysis Report

## Dataset Overview

- **Training Images**: 69863
- **Validation Images**: 10000
- **Total Training Objects**: 1286871
- **Total Validation Objects**: 185526

## Class Distribution

| Class | Train Count | Val Count | Train % | Val % |
|-------|-------------|-----------|---------|-------|
| bike | 7210 | 1007 | 0.56% | 0.54% |
| bus | 11672 | 1597 | 0.91% | 0.86% |
| car | 713211 | 102506 | 55.42% | 55.25% |
| motor | 3002 | 452 | 0.23% | 0.24% |
| person | 91349 | 13262 | 7.10% | 7.15% |
| rider | 4517 | 649 | 0.35% | 0.35% |
| traffic light | 186117 | 26885 | 14.46% | 14.49% |
| traffic sign | 239686 | 34908 | 18.63% | 18.82% |
| train | 136 | 15 | 0.01% | 0.01% |
| truck | 29971 | 4245 | 2.33% | 2.29% |

## Key Findings

### Class Imbalance
- Most common class: **car** (713211 instances)
- Least common class: **train** (136 instances)
- Imbalance ratio: **5244.20:1**

### Recommendations
1. Consider data augmentation for underrepresented classes
2. Use weighted loss functions to handle class imbalance
3. Monitor per-class performance during training
4. Apply techniques like focal loss for hard examples

## Generated Visualizations

- `class_distribution.png`: Class distribution comparison
- `bbox_statistics.png`: Bounding box statistics
- `attributes_distribution.png`: Weather, scene, and time distributions
- `occlusion_truncation.png`: Occlusion and truncation analysis
- `anomalies.csv`: Detected anomalies in the dataset
