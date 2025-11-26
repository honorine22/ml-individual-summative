#!/bin/bash
# Load testing script for FaultSense API
# Tests model performance under different load scenarios

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}╔══════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║        FaultSense Load Testing Suite         ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════╝${NC}"
echo ""

# Check if API is running
API_URL="${API_URL:-http://localhost:8000}"
echo -e "${YELLOW}Testing API at: $API_URL${NC}"

if ! curl -s "$API_URL/health" > /dev/null; then
    echo -e "${RED}❌ API is not accessible at $API_URL${NC}"
    echo -e "${YELLOW}For local testing:${NC}"
    echo "  ./scripts/start.sh"
    echo "  OR"
    echo "  docker-compose -f infra/docker-compose.yaml up"
    echo ""
    echo -e "${YELLOW}For cloud deployment testing:${NC}"
    echo "  export API_URL=https://your-app.railway.app"
    echo "  export API_URL=https://your-app.onrender.com"
    echo "  export API_URL=https://your-app.run.app"
    echo ""
    echo -e "${BLUE}💡 Make sure your deployed API is accessible and try again${NC}"
    exit 1
fi

echo -e "${GREEN}✅ API is running${NC}"
echo ""

# Install locust if not available
if ! command -v locust &> /dev/null; then
    echo -e "${YELLOW}Installing Locust...${NC}"
    pip install locust
fi

# Create results directory
RESULTS_DIR="reports/load_testing"
mkdir -p "$RESULTS_DIR"

echo -e "${BLUE}Starting load testing scenarios...${NC}"
echo ""

# Test 1: Light Load (10 users, 2 users/sec spawn rate)
echo -e "${YELLOW}🔸 Test 1: Light Load (10 users, 60 seconds)${NC}"
locust -f scripts/locust_load_test.py \
    --host="$API_URL" \
    --users=10 \
    --spawn-rate=2 \
    --run-time=60s \
    --headless \
    --html="$RESULTS_DIR/light_load_report.html" \
    --csv="$RESULTS_DIR/light_load" \
    --logfile="$RESULTS_DIR/light_load.log"

echo ""

# Test 2: Medium Load (50 users, 5 users/sec spawn rate)
echo -e "${YELLOW}🔸 Test 2: Medium Load (50 users, 90 seconds)${NC}"
locust -f scripts/locust_load_test.py \
    --host="$API_URL" \
    --users=50 \
    --spawn-rate=5 \
    --run-time=90s \
    --headless \
    --html="$RESULTS_DIR/medium_load_report.html" \
    --csv="$RESULTS_DIR/medium_load" \
    --logfile="$RESULTS_DIR/medium_load.log"

echo ""

# Test 3: Heavy Load (100 users, 10 users/sec spawn rate)
echo -e "${YELLOW}🔸 Test 3: Heavy Load (100 users, 120 seconds)${NC}"
locust -f scripts/locust_load_test.py \
    --host="$API_URL" \
    --users=100 \
    --spawn-rate=10 \
    --run-time=120s \
    --headless \
    --html="$RESULTS_DIR/heavy_load_report.html" \
    --csv="$RESULTS_DIR/heavy_load" \
    --logfile="$RESULTS_DIR/heavy_load.log"

echo ""

# Test 4: Stress Test (200 users, burst pattern)
echo -e "${YELLOW}🔸 Test 4: Stress Test (200 users, 60 seconds)${NC}"
locust -f scripts/locust_load_test.py \
    --host="$API_URL" \
    --users=200 \
    --spawn-rate=20 \
    --run-time=60s \
    --headless \
    --html="$RESULTS_DIR/stress_test_report.html" \
    --csv="$RESULTS_DIR/stress_test" \
    --logfile="$RESULTS_DIR/stress_test.log"

echo ""
echo -e "${GREEN}✅ Load testing completed!${NC}"
echo ""
echo -e "${BLUE}📊 Results saved to:${NC}"
echo "  - HTML Reports: $RESULTS_DIR/*.html"
echo "  - CSV Data: $RESULTS_DIR/*.csv"
echo "  - Logs: $RESULTS_DIR/*.log"
echo ""
echo -e "${YELLOW}📈 Key metrics to analyze:${NC}"
echo "  - Response times (avg, min, max, p95, p99)"
echo "  - Requests per second (RPS)"
echo "  - Failure rate"
echo "  - Concurrent user handling"
echo ""
echo -e "${BLUE}Open the HTML reports in your browser to view detailed results!${NC}"
