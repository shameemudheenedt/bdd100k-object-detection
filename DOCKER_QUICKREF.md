# Docker Quick Reference - BDD100K Project

## Quick Start (Copy & Paste)

### Build Image
```bash
docker build -t bdd100k-analysis .
```

### Run Data Analysis
```bash
docker run --rm -v $(pwd)/data:/app/data -v $(pwd)/analysis_output:/app/analysis_output bdd100k-analysis
```

### Run Dashboard
```bash
docker run --rm -p 8501:8501 -v $(pwd)/data:/app/data -v $(pwd)/analysis_output:/app/analysis_output bdd100k-analysis streamlit run dashboard.py --server.address=0.0.0.0
```
**Access**: http://localhost:8501

### Docker Compose
```bash
# Data analysis
docker-compose up data-analysis

# Dashboard (background)
docker-compose up -d dashboard

# Stop all
docker-compose down
```

## Command Cheat Sheet

| Task | Command |
|------|---------|
| Build base image | `docker build -t bdd100k-analysis .` |
| Build full image | `docker build -f Dockerfile.full -t bdd100k-full .` |
| Run analysis | `docker run --rm -v $(pwd)/data:/app/data -v $(pwd)/analysis_output:/app/analysis_output bdd100k-analysis` |
| Run dashboard | `docker run --rm -p 8501:8501 -v $(pwd)/data:/app/data -v $(pwd)/analysis_output:/app/analysis_output bdd100k-analysis streamlit run dashboard.py --server.address=0.0.0.0` |
| List images | `docker images \| grep bdd100k` |
| List containers | `docker ps -a` |
| Remove image | `docker rmi bdd100k-analysis` |
| Clean up | `docker system prune -a` |
| View logs | `docker logs <container-name>` |
| Test setup | `./test_docker.sh` |

## Verification Commands

```bash
# Check if image exists
docker images | grep bdd100k-analysis

# Verify analysis outputs
ls -lh analysis_output/*.csv

# Check container status
docker ps -a | grep bdd100k

# View container logs
docker logs bdd100k-analysis
```

##  Troubleshooting

| Problem | Solution |
|---------|----------|
| Port 8501 in use | `docker run -p 8502:8501 ...` |
| Permission denied | `chmod -R 755 data/ analysis_output/` |
| Out of space | `docker system prune -a` |
| Container won't start | `docker logs <container-name>` |
| Dashboard not loading | Run `data_analysis.py` first to generate CSVs |

##  Volume Mounts

| Host | Container | Purpose |
|------|-----------|---------|
| `./data` | `/app/data` | Dataset |
| `./analysis_output` | `/app/analysis_output` | Results |
| `./bdd100k_yolo` | `/app/bdd100k_yolo` | YOLO data |
| `./yolov5s_bdd100k` | `/app/yolov5s_bdd100k` | Model |

##  Expected Outputs

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

##  Documentation

- **README.md** - Main documentation
- **DOCKER_GUIDE.md** - Detailed Docker guide
- **DOCKER_TEST_SUMMARY.md** - Testing results
- **test_docker.sh** - Automated tests

##  For Reviewers

**Fastest way to test**:
```bash
# 1. Build
docker build -t bdd100k-analysis .

# 2. Run analysis
docker-compose up data-analysis

# 3. View dashboard
docker-compose up dashboard

# 4. Access http://localhost:8501
```

**Or use automated test**:
```bash
./test_docker.sh
```
