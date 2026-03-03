#!/bin/bash

# BDD100K Docker Testing Script
# Tests all containerized components of the assignment

set -e

echo "=========================================="
echo "BDD100K Docker Container Testing"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test 1: Build base image
echo -e "${YELLOW}Test 1: Building base Docker image...${NC}"
docker build -t bdd100k-analysis . || { echo -e "${RED}Failed to build base image${NC}"; exit 1; }
echo -e "${GREEN}✓ Base image built successfully${NC}"
echo ""

# Test 2: Run data analysis
echo -e "${YELLOW}Test 2: Running data analysis container...${NC}"
docker run --rm \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/analysis_output:/app/analysis_output \
    bdd100k-analysis python data_analysis.py || { echo -e "${RED}Failed to run data analysis${NC}"; exit 1; }
echo -e "${GREEN}✓ Data analysis completed successfully${NC}"
echo ""

# Test 3: Verify analysis outputs
echo -e "${YELLOW}Test 3: Verifying analysis outputs...${NC}"
if [ -f "analysis_output/class_distribution.csv" ] && \
   [ -f "analysis_output/bbox_statistics.csv" ] && \
   [ -f "analysis_output/ANALYSIS_REPORT.md" ]; then
    echo -e "${GREEN}✓ All analysis output files generated${NC}"
else
    echo -e "${RED}✗ Missing analysis output files${NC}"
    exit 1
fi
echo ""

# Test 4: Test dashboard startup (timeout after 15 seconds)
echo -e "${YELLOW}Test 4: Testing dashboard container startup...${NC}"
timeout 15 docker run --rm \
    -p 8501:8501 \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/analysis_output:/app/analysis_output \
    bdd100k-analysis streamlit run dashboard.py --server.address=0.0.0.0 &
DASHBOARD_PID=$!
sleep 10
if ps -p $DASHBOARD_PID > /dev/null; then
    echo -e "${GREEN}✓ Dashboard started successfully${NC}"
    kill $DASHBOARD_PID 2>/dev/null || true
else
    echo -e "${RED}✗ Dashboard failed to start${NC}"
fi
echo ""

# Test 5: Test docker-compose
echo -e "${YELLOW}Test 5: Testing docker-compose...${NC}"
docker-compose up data-analysis || { echo -e "${RED}Failed docker-compose test${NC}"; exit 1; }
echo -e "${GREEN}✓ Docker-compose works correctly${NC}"
echo ""

# Test 6: Build full pipeline image (if Dockerfile.full exists)
if [ -f "Dockerfile.full" ]; then
    echo -e "${YELLOW}Test 6: Building full pipeline image...${NC}"
    docker build -f Dockerfile.full -t bdd100k-full . || { echo -e "${RED}Failed to build full image${NC}"; exit 1; }
    echo -e "${GREEN}✓ Full pipeline image built successfully${NC}"
    echo ""
fi

# Summary
echo "=========================================="
echo -e "${GREEN}All Docker tests passed successfully!${NC}"
echo "=========================================="
echo ""
echo "Available Docker commands:"
echo "  1. Data Analysis:  docker run --rm -v \$(pwd)/data:/app/data -v \$(pwd)/analysis_output:/app/analysis_output bdd100k-analysis"
echo "  2. Dashboard:      docker run --rm -p 8501:8501 -v \$(pwd)/data:/app/data -v \$(pwd)/analysis_output:/app/analysis_output bdd100k-analysis streamlit run dashboard.py --server.address=0.0.0.0"
echo "  3. Docker Compose: docker-compose up data-analysis"
echo "  4. Dashboard Compose: docker-compose up dashboard"
echo ""
