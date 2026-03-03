# Docker Setup - Quick Start Guide

## Prerequisites
- Docker installed (20.10+)
- Docker Compose installed (1.29+)
- BDD100K dataset in `data/` directory

## Quick Start (3 Commands)

### 1. Build Docker Image
```bash
docker build -t bdd100k-analysis .
```

### 2. Run Data Analysis
```bash
docker run --rm \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/analysis_output:/app/analysis_output \
    bdd100k-analysis
```

### 3. View Dashboard
```bash
docker run --rm -p 8501:8501 \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/analysis_output:/app/analysis_output \
    bdd100k-analysis streamlit run dashboard.py --server.address=0.0.0.0
```
**Access**: http://localhost:8501

## Alternative: Docker Compose

```bash
# Run data analysis
docker-compose up data-analysis

# Run dashboard
docker-compose up dashboard
```

## Expected Output

### Data Analysis
```
Loading datasets...
Analyzing class distributions...
Analyzing bounding box statistics...
Analyzing dataset attributes...
Analyzing occlusion and truncation...
Analyzing objects per image...
Detecting anomalies...
Generating summary report...
Analysis complete! Results saved to analysis_output
```

### Dashboard
```
You can now view your Streamlit app in your browser.
URL: http://0.0.0.0:8501
```

## Generated Files

After running data analysis, check `analysis_output/`:
-  class_distribution.csv
-  bbox_statistics.csv
-  weather_distribution.csv
-  scene_distribution.csv
-  timeofday_distribution.csv
-  anomalies.csv
-  occlusion_truncation_stats.csv
-  objects_per_image_stats.csv
-  ANALYSIS_REPORT.md
-  PNG visualizations

##  Test Everything

```bash
./test_docker.sh
```

##  Full Documentation

- **DOCKER_GUIDE.md** - Comprehensive Docker guide
- **DOCKER_QUICKREF.md** - Command reference
- **DOCKER_TEST_SUMMARY.md** - Testing results
- **README.md** - Complete project documentation

##  Troubleshooting

| Problem | Solution |
|---------|----------|
| Port 8501 in use | Use `-p 8502:8501` instead |
| Permission denied | Run `chmod -R 755 data/ analysis_output/` |
| Out of disk space | Run `docker system prune -a` |
| Dashboard shows no data | Run data analysis first |

##  Need Help?

1. Check **DOCKER_GUIDE.md** for detailed instructions
2. Review error logs: `docker logs <container-name>`
3. Verify dataset structure matches README.md

---

**Status**:  Tested and Working  
**Last Updated**: March 3, 2025
