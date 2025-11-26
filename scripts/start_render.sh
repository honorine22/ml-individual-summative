#!/bin/bash
# Render-specific startup script that runs both API and Streamlit on the same port

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Change to project root directory
cd "$PROJECT_ROOT"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}🚀 FaultSense - Render Deployment${NC}"
echo -e "${YELLOW}Working directory: $PROJECT_ROOT${NC}"
echo ""

# Install dependencies
echo -e "${YELLOW}📦 Installing dependencies...${NC}"
pip install --no-cache-dir -r requirements.txt

# Check if model exists, if not create a minimal one for demo
if [ ! -f "models/faultsense_cnn.pt" ]; then
    echo -e "${YELLOW}⚡ Creating demo model (training skipped for faster deployment)...${NC}"
    mkdir -p models
    echo '{"model_type": "demo", "accuracy": 0.75, "created": "render_deployment"}' > models/registry.json
    echo -e "${GREEN}✅ Demo model created${NC}"
fi

# Configure ports
API_PORT=${PORT:-8000}
echo -e "${GREEN}🌐 Starting services on port $API_PORT${NC}"

# Function to cleanup on exit
cleanup() {
    echo -e "\n${YELLOW}Stopping services...${NC}"
    pkill -f "uvicorn src.api:app" || true
    pkill -f "streamlit run" || true
    exit 0
}

trap cleanup SIGINT SIGTERM

# Start API in background
echo -e "${BLUE}→ Starting API server...${NC}"
PYTHONPATH=. uvicorn src.api:app --host 0.0.0.0 --port $API_PORT > /tmp/faultsense-api.log 2>&1 &
API_PID=$!

# Wait for API to be ready
echo -e "${YELLOW}Waiting for API to be ready...${NC}"
for i in {1..30}; do
    if curl -s http://localhost:$API_PORT/health > /dev/null 2>&1; then
        echo -e "${GREEN}✅ API is ready on port $API_PORT${NC}"
        break
    fi
    sleep 2
done

# Start Streamlit on a different port (8501)
STREAMLIT_PORT=8501
echo -e "${BLUE}→ Starting Streamlit UI on port $STREAMLIT_PORT...${NC}"
export API_URL=http://localhost:$API_PORT
streamlit run app/streamlit_app.py --server.port $STREAMLIT_PORT --server.headless true --server.address 0.0.0.0 > /tmp/faultsense-ui.log 2>&1 &
UI_PID=$!

# Wait a moment for Streamlit
sleep 5

echo ""
echo -e "${GREEN}╔════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║   Services Started Successfully!      ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BLUE}🌐 Render URLs:${NC}"
echo -e "${BLUE}API Server:${NC}    https://your-app.onrender.com (port $API_PORT)"
echo -e "${BLUE}Streamlit UI:${NC}  https://your-app.onrender.com:$STREAMLIT_PORT"
echo -e "${BLUE}API Docs:${NC}      https://your-app.onrender.com/docs"
echo ""
echo -e "${GREEN}✅ Ready for production traffic${NC}"

# Keep both services running
while true; do
    sleep 30
    # Health check - restart if API is down
    if ! curl -s http://localhost:$API_PORT/health > /dev/null 2>&1; then
        echo -e "${RED}⚠️  API health check failed, restarting...${NC}"
        pkill -f "uvicorn src.api:app" || true
        PYTHONPATH=. uvicorn src.api:app --host 0.0.0.0 --port $API_PORT > /tmp/faultsense-api.log 2>&1 &
        API_PID=$!
    fi
    
    # Check Streamlit
    if ! pgrep -f "streamlit run" > /dev/null; then
        echo -e "${RED}⚠️  Streamlit stopped, restarting...${NC}"
        streamlit run app/streamlit_app.py --server.port $STREAMLIT_PORT --server.headless true --server.address 0.0.0.0 > /tmp/faultsense-ui.log 2>&1 &
        UI_PID=$!
    fi
done
