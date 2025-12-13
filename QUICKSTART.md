# Quick Start Guide

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Option 1: Automatic Installation (Recommended)

```bash
./install.sh
```

### Option 2: Manual Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .

# Verify installation
python test_installation.py
```

## Running the Pipeline

### Process Small Boxes (All Tasks)

```bash
python -m smart_palletizier.pipeline --data-folder data/small_box --box-type small_box
```

### Process Medium Boxes

```bash
python -m smart_palletizier.pipeline --data-folder data/medium_box --box-type medium_box
```

### Run Specific Tasks

```bash
# Run only Task 1 and Task 4
python -m smart_palletizier.pipeline --data-folder data/small_box --tasks 1,4
```

## What Each Task Does

- **Task 1**: Detects boxes in 2D images using contour detection
- **Task 2**: Detects planar patches (box faces) in 3D point clouds
- **Task 3**: Cleans noisy point clouds while preserving dimensions
- **Task 4**: Estimates 6D poses (position + orientation) of boxes

## Documentation

- **[METHODOLOGY.md](METHODOLOGY.md)** - Detailed explanation of approaches
- **[USAGE.md](USAGE.md)** - Complete API usage guide
- **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)** - Project overview

## Example Usage

```python
from smart_palletizier import BoxDetector, PoseEstimator
import cv2

# Detect boxes in image
detector = BoxDetector()
image = cv2.imread("data/small_box/color_image.png")
boxes = detector.detect(image)

# Estimate pose
pose_estimator = PoseEstimator()
# ... (see USAGE.md for full examples)
```

## Results

The pipeline produces:
- 2D detection images with bounding boxes
- 3D visualizations of planar patches
- Cleaned point cloud files
- Pose estimation results (position, orientation)

## Getting Help

- Check [USAGE.md](USAGE.md) for detailed examples
- Read [METHODOLOGY.md](METHODOLOGY.md) for technical details
- Run `python test_installation.py` to verify setup

## Key Features

✅ No deep learning frameworks required (no PyTorch, TensorFlow)  
✅ Fast CPU-based processing  
✅ Uses classical CV methods (ICP, RANSAC, contour detection)  
✅ Well-documented code with examples  
✅ Modular design for easy extension
