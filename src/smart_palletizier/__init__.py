"""
Smart Palletizer Package

A computer vision toolkit for box detection and 6D pose estimation in palletizing applications.

This package provides classical computer vision methods for:
- 2D box detection in images
- Planar surface segmentation in point clouds
- Point cloud cleaning and filtering
- 6D pose estimation using ICP registration

Author: Smart Palletizer Team
License: MIT
"""

__version__ = "0.1.0"
__author__ = "Smart Palletizer Team"

from .box_detector import BoxDetector
from .pointcloud_processor import PointCloudProcessor
from .planar_detector import PlanarPatchDetector
from .pose_estimator import PoseEstimator
from .visualization import Visualizer

__all__ = [
    'BoxDetector',
    'PointCloudProcessor',
    'PlanarPatchDetector',
    'PoseEstimator',
    'Visualizer',
]
