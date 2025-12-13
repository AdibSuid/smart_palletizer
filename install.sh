#!/bin/bash

# Smart Palletizer - Installation and Test Script

echo "=========================================="
echo "Smart Palletizer - Installation Script"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python3 --version

# Create virtual environment (optional)
read -p "Create virtual environment? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    echo "Virtual environment activated"
fi

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

# Install package in development mode
echo ""
echo "Installing smart_palletizer package..."
pip install -e .

# Verify installation
echo ""
echo "Verifying installation..."
python3 -c "from smart_palletizier import BoxDetector, PointCloudProcessor, PoseEstimator; print('✓ Package imported successfully')"

echo ""
echo "=========================================="
echo "Installation completed!"
echo "=========================================="
echo ""
echo "To run the pipeline:"
echo "  python -m smart_palletizier.pipeline --data-folder data/small_box --box-type small_box"
echo ""
echo "To run examples:"
echo "  python examples/demo.py"
echo ""
