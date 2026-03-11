#!/bin/bash

echo "================================================"
echo "   Expense Classification System - Setup"
echo "================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 is not installed${NC}"
    echo "Please install Python 3.8 or higher and try again"
    exit 1
fi

echo -e "${GREEN}✓ Python 3 found${NC}"
python3 --version
echo ""

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo -e "${RED}Error: pip3 is not installed${NC}"
    echo "Please install pip and try again"
    exit 1
fi

echo -e "${GREEN}✓ pip found${NC}"
echo ""

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${GREEN}✓ Virtual environment exists${NC}"
fi
echo ""

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate
echo -e "${GREEN}✓ Virtual environment activated${NC}"
echo ""

# Install dependencies
echo -e "${YELLOW}Installing dependencies...${NC}"
pip install --upgrade pip --break-system-packages 2>/dev/null || pip install --upgrade pip
pip install -r requirements.txt --break-system-packages 2>/dev/null || pip install -r requirements.txt
echo -e "${GREEN}✓ Dependencies installed${NC}"
echo ""

# Create necessary directories
echo -e "${YELLOW}Creating directories...${NC}"
mkdir -p uploads outputs data templates
echo -e "${GREEN}✓ Directories created${NC}"
echo ""

# ── API Keys ─────────────────────────────────────────────────────
# Keys are loaded from a .env file — NEVER hard-code them here.
# Copy .env.example to .env and fill in your key:
#   cp .env.example .env && nano .env
if [ -f ".env" ]; then
    set -a; source .env; set +a
    echo -e "${GREEN}✓ .env file loaded${NC}"
    echo ""
else
    echo -e "${YELLOW}⚠️  No .env file found${NC}"
    echo "   Create one from the template:  cp .env.example .env"
    echo "   Then add your OPENROUTER_API_KEY inside it."
    echo ""
fi

if [ -n "$OPENROUTER_API_KEY" ]; then
    echo -e "${GREEN}✓ OPENROUTER_API_KEY detected — AI classification enabled${NC}"
    echo ""
else
    echo -e "${YELLOW}⚠️  OPENROUTER_API_KEY not set — keyword-only mode will be used${NC}"
    echo "   Get a free key at https://openrouter.ai and add it to .env"
    echo ""
fi

echo "================================================"
echo -e "${GREEN}   Setup Complete!${NC}"
echo "================================================"
echo ""
echo "Starting the application..."
echo ""
echo "📊 Web Interface: http://localhost:5000"
echo ""
echo "Sample files included:"
echo "  • sample_categories.csv - Example expense categories"
echo "  • sample_transactions.csv - Example bank transactions"
echo ""
echo "To stop the server, press Ctrl+C"
echo ""
echo "================================================"
echo ""

# Run the Flask app
python3 app.py