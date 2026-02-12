#!/bin/bash

# Propensity Modeling - Quick Start Script
# Runs the complete end-to-end pipeline

echo "=========================================="
echo "Propensity Modeling - Quick Start"
echo "=========================================="

# Check Python version
python_version=$(python3 --version 2>&1)
echo "Python version: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

echo ""
echo "=========================================="
echo "Running End-to-End Demo"
echo "=========================================="
echo ""

# Run demo
python demo.py

echo ""
echo "=========================================="
echo "Demo Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Start prediction API:"
echo "   uvicorn src.serving.prediction_service:app --reload"
echo ""
echo "2. View plots in: plots/"
echo "3. Check models in: models/"
echo "4. Review data in: data/"
echo ""
echo "For batch scoring:"
echo "   python src/serving/batch_scoring.py --help"
echo ""
