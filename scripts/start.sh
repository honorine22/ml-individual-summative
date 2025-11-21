#!/bin/bash
# Modern startup script for FaultSense

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   FaultSense MLOps Platform          ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════╝${NC}"
echo ""

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv .venv
fi

# Activate virtual environment
source .venv/bin/activate

# Install dependencies if needed
if [ ! -f ".venv/.installed" ]; then
    echo -e "${YELLOW}Installing dependencies...${NC}"
    pip install -q -r requirements.txt
    touch .venv/.installed
fi

# Check if model exists
if [ ! -f "models/faultsense_cnn.pt" ]; then
    echo -e "${YELLOW}Model not found. Training initial model...${NC}"
    PYTHONPATH=. python scripts/run_pipeline.py --stage train --epochs 20
fi

# Start services
echo -e "${GREEN}Starting services...${NC}"
echo ""

# Function to cleanup on exit
cleanup() {
    echo -e "\n${YELLOW}Stopping services...${NC}"
    pkill -f "uvicorn src.api:app" || true
    pkill -f "streamlit run" || true
    exit 0
}

trap cleanup SIGINT SIGTERM

# Start API in background
echo -e "${BLUE}→ Starting API server on http://localhost:8000${NC}"
PYTHONPATH=. uvicorn src.api:app --host 0.0.0.0 --port 8000 > /tmp/faultsense-api.log 2>&1 &
API_PID=$!

# Wait for API to be ready
echo -e "${YELLOW}Waiting for API to be ready...${NC}"
for i in {1..30}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo -e "${GREEN}✓ API is ready!${NC}"
        break
    fi
    sleep 1
done

# Start Streamlit
echo -e "${BLUE}→ Starting Streamlit UI on http://localhost:8501${NC}"
export API_URL=http://localhost:8000
streamlit run app/streamlit_app.py --server.port 8501 --server.headless true > /tmp/faultsense-ui.log 2>&1 &
UI_PID=$!

# Wait a moment for Streamlit
sleep 3

echo ""
echo -e "${GREEN}╔════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║   Services Started Successfully!      ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BLUE}API Server:${NC}    http://localhost:8000"
echo -e "${BLUE}Streamlit UI:${NC}  http://localhost:8501"
echo -e "${BLUE}API Docs:${NC}      http://localhost:8000/docs"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop all services${NC}"
echo ""

# Wait for user interrupt
wait

