#!/bin/bash
# Local setup script for FaultSense development

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}🔧 FaultSense Local Setup${NC}"
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Change to project root directory
cd "$PROJECT_ROOT"

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv .venv
fi

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source .venv/bin/activate

# Install dependencies
echo -e "${YELLOW}Installing dependencies...${NC}"
pip install --upgrade pip
pip install -r requirements.txt

# Create demo model
echo -e "${YELLOW}Creating demo model...${NC}"
PYTHONPATH=. python scripts/create_demo_model.py

echo ""
echo -e "${GREEN}✅ Local setup complete!${NC}"
echo ""
echo -e "${BLUE}To start the services:${NC}"
echo "  ./scripts/start.sh"
echo ""
echo -e "${BLUE}Or manually:${NC}"
echo "  source .venv/bin/activate"
echo "  PYTHONPATH=. uvicorn src.api:app --reload --port 8000"
