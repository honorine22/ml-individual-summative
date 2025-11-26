#!/bin/bash
<<<<<<< HEAD
# Modern startup script for FaultSense - supports local and deployment environments

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Change to project root directory
cd "$PROJECT_ROOT"

=======
# Modern startup script for FaultSense

set -e

>>>>>>> f089888ffe051bc4672a0ce2d8109c6f0c4006cf
# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
<<<<<<< HEAD
RED='\033[0;31m'
=======
>>>>>>> f089888ffe051bc4672a0ce2d8109c6f0c4006cf
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   FaultSense MLOps Platform          ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════╝${NC}"
<<<<<<< HEAD
echo -e "${YELLOW}Working directory: $PROJECT_ROOT${NC}"
echo ""

# Check if we're in a deployment environment
DEPLOYMENT_ENV=${DEPLOYMENT_ENV:-"false"}
if [[ "$RAILWAY_ENVIRONMENT" == "production" ]] || [[ "$RENDER" == "true" ]] || [[ "$GAE_ENV" == "standard" ]] || [[ "$DEPLOYMENT_ENV" == "true" ]]; then
    DEPLOYMENT_ENV="true"
    echo -e "${GREEN}🚀 Running in deployment environment${NC}"
else
    echo -e "${BLUE}🏠 Running in local environment${NC}"
fi

# Handle dependencies based on environment
if [[ "$DEPLOYMENT_ENV" == "true" ]]; then
    echo -e "${GREEN}📦 Deployment environment - dependencies should be pre-installed${NC}"
    # In deployment, dependencies are usually installed during build
    if [ -f "requirements.txt" ]; then
        echo -e "${YELLOW}Ensuring dependencies are up to date...${NC}"
        pip install -q -r requirements.txt || echo -e "${YELLOW}⚠️  Some dependencies may already be installed${NC}"
    fi
else
    # Local development setup
    if [ ! -d ".venv" ]; then
        echo -e "${YELLOW}Creating virtual environment...${NC}"
        python3 -m venv .venv
    fi

    # Activate virtual environment
    source .venv/bin/activate

    # Install dependencies if needed
    if [ ! -f ".venv/.installed" ] || [ ! -f "requirements.txt" ]; then
        if [ -f "requirements.txt" ]; then
            echo -e "${YELLOW}Installing dependencies...${NC}"
            pip install -q -r requirements.txt
            touch .venv/.installed
        else
            echo -e "${RED}❌ requirements.txt not found in $PROJECT_ROOT${NC}"
            echo -e "${YELLOW}Please ensure you're running this script from the project root or that requirements.txt exists${NC}"
            exit 1
        fi
    fi
=======
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
>>>>>>> f089888ffe051bc4672a0ce2d8109c6f0c4006cf
fi

# Check if model exists
if [ ! -f "models/faultsense_cnn.pt" ]; then
    echo -e "${YELLOW}Model not found. Training initial model...${NC}"
<<<<<<< HEAD
    if [[ "$DEPLOYMENT_ENV" == "true" ]]; then
        echo -e "${YELLOW}🚀 Deployment environment - training with minimal epochs for faster startup${NC}"
        PYTHONPATH=. python scripts/run_pipeline.py --stage train --epochs 10
    else
        echo -e "${BLUE}🏠 Local environment - training with standard epochs${NC}"
        PYTHONPATH=. python scripts/run_pipeline.py --stage train --epochs 20
    fi
=======
    PYTHONPATH=. python scripts/run_pipeline.py --stage train --epochs 20
>>>>>>> f089888ffe051bc4672a0ce2d8109c6f0c4006cf
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

<<<<<<< HEAD
# Configure ports based on environment
if [[ "$DEPLOYMENT_ENV" == "true" ]]; then
    API_PORT=${PORT:-8000}
    STREAMLIT_PORT=${STREAMLIT_PORT:-8501}
    API_HOST="0.0.0.0"
    echo -e "${GREEN}🌐 Deployment mode - API on port $API_PORT, Streamlit on port $STREAMLIT_PORT${NC}"
else
    API_PORT=8000
    STREAMLIT_PORT=8501
    API_HOST="0.0.0.0"
    echo -e "${BLUE}🏠 Local mode - API on port $API_PORT, Streamlit on port $STREAMLIT_PORT${NC}"
fi

# Start API in background
echo -e "${BLUE}→ Starting API server on http://$API_HOST:$API_PORT${NC}"
PYTHONPATH=. uvicorn src.api:app --host $API_HOST --port $API_PORT > /tmp/faultsense-api.log 2>&1 &
=======
# Start API in background
echo -e "${BLUE}→ Starting API server on http://localhost:8000${NC}"
PYTHONPATH=. uvicorn src.api:app --host 0.0.0.0 --port 8000 > /tmp/faultsense-api.log 2>&1 &
>>>>>>> f089888ffe051bc4672a0ce2d8109c6f0c4006cf
API_PID=$!

# Wait for API to be ready
echo -e "${YELLOW}Waiting for API to be ready...${NC}"
for i in {1..30}; do
<<<<<<< HEAD
    if curl -s http://localhost:$API_PORT/health > /dev/null 2>&1; then
=======
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
>>>>>>> f089888ffe051bc4672a0ce2d8109c6f0c4006cf
        echo -e "${GREEN}✓ API is ready!${NC}"
        break
    fi
    sleep 1
done

<<<<<<< HEAD
# Start Streamlit (only if not in API-only deployment)
if [[ "$API_ONLY" != "true" ]]; then
    echo -e "${BLUE}→ Starting Streamlit UI on http://$API_HOST:$STREAMLIT_PORT${NC}"
    export API_URL=http://localhost:$API_PORT
    streamlit run app/streamlit_app.py --server.port $STREAMLIT_PORT --server.headless true --server.address $API_HOST > /tmp/faultsense-ui.log 2>&1 &
    UI_PID=$!
else
    echo -e "${YELLOW}🔧 API-only mode - Streamlit not started${NC}"
fi
=======
# Start Streamlit
echo -e "${BLUE}→ Starting Streamlit UI on http://localhost:8501${NC}"
export API_URL=http://localhost:8000
streamlit run app/streamlit_app.py --server.port 8501 --server.headless true > /tmp/faultsense-ui.log 2>&1 &
UI_PID=$!
>>>>>>> f089888ffe051bc4672a0ce2d8109c6f0c4006cf

# Wait a moment for Streamlit
sleep 3

echo ""
echo -e "${GREEN}╔════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║   Services Started Successfully!      ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════╝${NC}"
echo ""
<<<<<<< HEAD

if [[ "$DEPLOYMENT_ENV" == "true" ]]; then
    echo -e "${BLUE}🚀 Deployment URLs:${NC}"
    echo -e "${BLUE}API Server:${NC}    http://0.0.0.0:$API_PORT"
    echo -e "${BLUE}API Docs:${NC}      http://0.0.0.0:$API_PORT/docs"
    if [[ "$API_ONLY" != "true" ]]; then
        echo -e "${BLUE}Streamlit UI:${NC}  http://0.0.0.0:$STREAMLIT_PORT"
    fi
    echo -e "${GREEN}✅ Ready for production traffic${NC}"
else
    echo -e "${BLUE}🏠 Local URLs:${NC}"
    echo -e "${BLUE}API Server:${NC}    http://localhost:$API_PORT"
    echo -e "${BLUE}Streamlit UI:${NC}  http://localhost:$STREAMLIT_PORT"
    echo -e "${BLUE}API Docs:${NC}      http://localhost:$API_PORT/docs"
    echo ""
    echo -e "${YELLOW}Press Ctrl+C to stop all services${NC}"
fi
echo ""

# In deployment, keep running; locally, wait for interrupt
if [[ "$DEPLOYMENT_ENV" == "true" ]]; then
    # Keep the process alive in deployment
    while true; do
        sleep 30
        # Health check - restart if API is down
        if ! curl -s http://localhost:$API_PORT/health > /dev/null 2>&1; then
            echo -e "${RED}⚠️  API health check failed, restarting...${NC}"
            pkill -f "uvicorn src.api:app" || true
            PYTHONPATH=. uvicorn src.api:app --host $API_HOST --port $API_PORT > /tmp/faultsense-api.log 2>&1 &
            API_PID=$!
        fi
    done
else
    # Wait for user interrupt in local mode
    wait
fi
=======
echo -e "${BLUE}API Server:${NC}    http://localhost:8000"
echo -e "${BLUE}Streamlit UI:${NC}  http://localhost:8501"
echo -e "${BLUE}API Docs:${NC}      http://localhost:8000/docs"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop all services${NC}"
echo ""

# Wait for user interrupt
wait
>>>>>>> f089888ffe051bc4672a0ce2d8109c6f0c4006cf

