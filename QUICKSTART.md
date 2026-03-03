# Quick Start Guide

## Prerequisites
- Python 3.9+
- Docker (optional)
- CUDA-capable GPU (optional, for training)

## Step 1: Data Analysis (Required)

### Option A: Local
```bash
# Install dependencies
pip install -r requirements.txt

# Run analysis
python data_analysis.py

# View results
ls analysis_output/
```

### Option B: Docker
```bash
# Build and run
docker build -t bdd100k-analysis .
docker run -v $(pwd)/data:/app/data \
           -v $(pwd)/analysis_output:/app/analysis_output \
           bdd100k-analysis

# Or use docker-compose
docker-compose up data-analysis
```

**Expected Output:**
- `analysis_output/` directory with CSV files and PNG visualizations
- `ANALYSIS_REPORT.md` with comprehensive findings

## Step 2: Interactive Dashboard (Optional)

```bash
# Local
streamlit run dashboard.py

# Docker
docker-compose up dashboard
# Access at http://localhost:8501
```

## Step 3: Model Preparation

```bash
# Convert dataset to YOLO format
python model_pipeline.py --task prepare

# This creates bdd100k_yolo/ directory
```

## Step 4: Model Training (Optional)

```bash
# Quick test (1 epoch)
python model_pipeline.py --task train --epochs 1

# Full training
cd yolov5s_bdd100k
python train.py --data ../bdd100k_yolo/bdd100k.yaml \
                --cfg models/custom_yolov5s.yaml \
                --weights weights/yolov5s.pt \
                --epochs 50 \
                --batch-size 16
```

## Step 5: Inference

```bash
# Single image
python inference.py --weights runs/train/bdd100k_exp/weights/best.pt \
                    --source data/bdd100k_images_100k/bdd100k/images/100k/val/sample.jpg

# Batch inference
python inference.py --weights runs/train/bdd100k_exp/weights/best.pt \
                    --source data/bdd100k_images_100k/bdd100k/images/100k/val/ \
                    --output predictions.json
```

## Step 6: Evaluation

```bash
# Run evaluation
python evaluation.py

# View results
ls evaluation_output/
ls visualizations/
```

## Troubleshooting

### Issue: Out of memory during analysis
**Solution:** Process subset of data or increase swap space

### Issue: Docker build fails
**Solution:** Ensure Docker has sufficient resources (4GB+ RAM)

### Issue: CUDA not available
**Solution:** Training will use CPU (slower). Use `--device cpu` flag

## File Checklist

After running all steps, you should have:
-  `analysis_output/` - Data analysis results
-  `bdd100k_yolo/` - Converted dataset
-  `runs/train/` - Training logs and weights
-  `predictions.json` - Model predictions
-  `evaluation_output/` - Evaluation metrics
-  `visualizations/` - Qualitative results

## Time Estimates

- Data Analysis: 10-15 minutes
- Dataset Conversion: 20-30 minutes
- Training (1 epoch): 30-45 minutes
- Training (50 epochs): 24-36 hours
- Evaluation: 15-20 minutes

## Next Steps

1. Review `ANALYSIS_REPORT.md` for dataset insights
2. Check dashboard for interactive exploration
3. Analyze `evaluation_output/` for model performance
4. Review `visualizations/` for failure cases
5. Iterate on model improvements
