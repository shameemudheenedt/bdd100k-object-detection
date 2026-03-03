# Docker Usage Guide - BDD100K Object Detection Project

This guide provides complete instructions for running the BDD100K object detection project using Docker containers.

## Prerequisites

- Docker installed (version 20.10+)
- Docker Compose installed (version 1.29+)
- NVIDIA Docker runtime (for GPU training, optional)
- BDD100K dataset downloaded in `data/` directory

## Quick Start

### 1. Build Docker Image

```bash
# Build the base image for data analysis
docker build -t bdd100k-analysis .

# Build the full pipeline image (includes model training/evaluation)
docker build -f Dockerfile.full -t bdd100k-full .
```

### 2. Run Data Analysis (Task 1)

```bash
# Run data analysis
docker run --rm \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/analysis_output:/app/analysis_output \
    bdd100k-analysis

# Launch interactive dashboard
docker run --rm \
    -p 8501:8501 \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/analysis_output:/app/analysis_output \
    bdd100k-analysis streamlit run dashboard.py --server.address=0.0.0.0

# Access dashboard at: http://localhost:8501
```

### 3. Using Docker Compose

```bash
# Run data analysis
docker-compose up data-analysis

# Run dashboard (in background)
docker-compose up -d dashboard

# Stop all services
docker-compose down
```

## Detailed Usage

### Task 1: Data Analysis

#### Option A: Direct Docker Command

```bash
docker run --rm \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/analysis_output:/app/analysis_output \
    -e DATA_DIR=/app/data \
    bdd100k-analysis python data_analysis.py
```

**Output:** Analysis results saved to `analysis_output/` directory:
- `class_distribution.csv`
- `bbox_statistics.csv`
- `weather_distribution.csv`
- `scene_distribution.csv`
- `timeofday_distribution.csv`
- `anomalies.csv`
- `ANALYSIS_REPORT.md`
- Various PNG visualizations

#### Option B: Docker Compose

```bash
docker-compose up data-analysis
```

#### Interactive Dashboard

```bash
# Using Docker
docker run --rm -p 8501:8501 \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/analysis_output:/app/analysis_output \
    bdd100k-analysis streamlit run dashboard.py --server.address=0.0.0.0

# Using Docker Compose
docker-compose up dashboard

# Access at: http://localhost:8501
```

### Task 2: Model Pipeline

#### Prepare Dataset (Convert to YOLO format)

```bash
docker run --rm \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/bdd100k_yolo:/app/bdd100k_yolo \
    bdd100k-full python model_pipeline.py --task prepare
```

#### Train Model (Requires GPU)

```bash
# With NVIDIA Docker runtime
docker run --rm --gpus all \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/bdd100k_yolo:/app/bdd100k_yolo \
    -v $(pwd)/yolov5s_bdd100k:/app/yolov5s_bdd100k \
    -v $(pwd)/runs:/app/runs \
    bdd100k-full bash -c "python fix_yolov5_pytorch26.py && python model_pipeline.py --task train --epochs 1"

# Using Docker Compose (with GPU)
docker-compose -f docker-compose.full.yml up model-train
```

**Note:** Training requires NVIDIA GPU and nvidia-docker2 runtime.

#### CPU-Only Training (Slower)

```bash
docker run --rm \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/bdd100k_yolo:/app/bdd100k_yolo \
    -v $(pwd)/yolov5s_bdd100k:/app/yolov5s_bdd100k \
    bdd100k-full bash -c "python fix_yolov5_pytorch26.py && python model_pipeline.py --task train --epochs 1 --device cpu"
```

### Task 3: Evaluation and Visualization

#### Run Evaluation

```bash
docker run --rm \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/bdd100k_yolo:/app/bdd100k_yolo \
    -v $(pwd)/yolov5s_bdd100k:/app/yolov5s_bdd100k \
    -v $(pwd)/evaluation_output:/app/evaluation_output \
    -v $(pwd)/visualizations:/app/visualizations \
    bdd100k-full python evaluation.py
```

#### Run Inference on Test Set

```bash
docker run --rm \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/yolov5s_bdd100k:/app/yolov5s_bdd100k \
    -v $(pwd)/high_conf_output:/app/high_conf_output \
    bdd100k-full python inference.py \
        --weights yolov5s_bdd100k/weights/yolov5s.pt \
        --source /app/data/bdd100k_images_100k/bdd100k/images/100k/test \
        --output high_conf_predictions.json \
        --save-img \
        --output-dir high_conf_output \
        --conf 0.5
```

## Complete Pipeline Execution

Run the entire pipeline using Docker Compose:

```bash
# Step 1: Data Analysis
docker-compose up data-analysis

# Step 2: View Dashboard (optional)
docker-compose up -d dashboard

# Step 3: Prepare Dataset
docker-compose -f docker-compose.full.yml up model-prepare

# Step 4: Train Model (requires GPU)
docker-compose -f docker-compose.full.yml up model-train

# Step 5: Evaluate Model
docker-compose -f docker-compose.full.yml up model-eval

# Step 6: Run Inference
docker-compose -f docker-compose.full.yml up model-inference

# Cleanup
docker-compose down
docker-compose -f docker-compose.full.yml down
```

## Volume Mounts Explained

| Host Path | Container Path | Purpose |
|-----------|---------------|---------|
| `./data` | `/app/data` | BDD100K dataset (read-only) |
| `./analysis_output` | `/app/analysis_output` | Data analysis results |
| `./bdd100k_yolo` | `/app/bdd100k_yolo` | YOLO format dataset |
| `./yolov5s_bdd100k` | `/app/yolov5s_bdd100k` | YOLOv5 model files |
| `./runs` | `/app/runs` | Training runs and logs |
| `./evaluation_output` | `/app/evaluation_output` | Evaluation metrics |
| `./visualizations` | `/app/visualizations` | Visualization images |
| `./high_conf_output` | `/app/high_conf_output` | Inference results |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATA_DIR` | `/app/data` | Base directory for dataset |
| `PYTHONUNBUFFERED` | `1` | Enable Python output buffering |

## Troubleshooting

### Issue: "Cannot connect to Docker daemon"
```bash
# Start Docker service
sudo systemctl start docker

# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker
```

### Issue: "Port 8501 already in use"
```bash
# Find and kill process using port
sudo lsof -ti:8501 | xargs kill -9

# Or use different port
docker run -p 8502:8501 ... bdd100k-analysis streamlit run dashboard.py
```

### Issue: "No space left on device"
```bash
# Clean up Docker
docker system prune -a

# Remove unused volumes
docker volume prune
```

### Issue: "NVIDIA runtime not found"
```bash
# Install nvidia-docker2
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

### Issue: "Permission denied" on volume mounts
```bash
# Fix permissions
chmod -R 755 data/ analysis_output/ bdd100k_yolo/
```

## Testing Docker Setup

Run the automated test script:

```bash
./test_docker.sh
```

This will:
1. Build Docker images
2. Run data analysis
3. Verify outputs
4. Test dashboard startup
5. Test docker-compose

## Performance Considerations

### CPU vs GPU
- **Data Analysis**: CPU sufficient (~5-10 minutes)
- **Model Training**: GPU recommended (1 epoch: ~30 min GPU vs 4+ hours CPU)
- **Inference**: GPU recommended for large test sets

### Memory Requirements
- **Data Analysis**: 4GB RAM minimum
- **Model Training**: 8GB RAM + 6GB VRAM (GPU)
- **Dashboard**: 2GB RAM

### Disk Space
- Docker images: ~5GB
- Dataset: ~5.3GB
- Model weights: ~15MB
- Analysis outputs: ~50MB
- Training outputs: ~500MB

## Best Practices

1. **Always run data analysis first** to generate CSV files for dashboard
2. **Use volume mounts** to persist data between container runs
3. **Clean up containers** after use with `--rm` flag
4. **Use docker-compose** for complex multi-step workflows
5. **Monitor resources** with `docker stats`

## Additional Commands

```bash
# View running containers
docker ps

# View all containers (including stopped)
docker ps -a

# View logs
docker logs bdd100k-analysis

# Execute command in running container
docker exec -it bdd100k-dashboard bash

# Remove all stopped containers
docker container prune

# Remove all unused images
docker image prune -a

# View disk usage
docker system df
```

## Support

For issues or questions:
1. Check this documentation
2. Review error logs: `docker logs <container-name>`
3. Verify dataset structure matches README
4. Ensure sufficient disk space and memory
