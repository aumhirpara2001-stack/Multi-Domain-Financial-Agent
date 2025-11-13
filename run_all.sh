#!/bin/bash
# PennyBot Setup and Run Script
# This script sets up the environment and runs the PennyBot RAG system

set -e  # Exit on error

echo "=================================================="
echo "PennyBot LLM Agentic RAG - Setup Script"
echo "=================================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.10+ first."
    exit 1
fi

echo "✅ Python 3 found: $(python3 --version)"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip --quiet

# Install requirements
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt --quiet
echo "✅ Dependencies installed"

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚠️  No .env file found!"
    echo "Please copy config/.env.example to .env and add your API keys:"
    echo "  cp config/.env.example .env"
    echo "  # Then edit .env with your API keys"
    exit 1
fi

echo "✅ .env file found"

# Run ingestion and filtering
echo ""
echo "=================================================="
echo "Step 1: Processing raw data..."
echo "=================================================="
python scripts/ingest_and_filter.py

# Build index
echo ""
echo "=================================================="
echo "Step 2: Building Pinecone index..."
echo "=================================================="
python scripts/build_index.py

# Launch chat CLI
echo ""
echo "=================================================="
echo "Step 3: Launching PennyBot Chat CLI..."
echo "=================================================="
python src/chat_cli.py

echo ""
echo "=================================================="
echo "All tasks completed successfully!"
echo "=================================================="
