"""
Test script to verify the smart_palletizer installation.
"""

import sys


def test_imports():
    """Test if all modules can be imported."""
    print("Testing imports...")
    
    try:
        import numpy as np
        print("✓ NumPy")
    except ImportError as e:
        print(f"✗ NumPy: {e}")
        return False
    
    try:
        import cv2
        print("✓ OpenCV")
    except ImportError as e:
        print(f"✗ OpenCV: {e}")
        return False
    
    try:
        import open3d as o3d
        print("✓ Open3D")
    except ImportError as e:
        print(f"✗ Open3D: {e}")
        return False
    
    try:
        import scipy
        print("✓ SciPy")
    except ImportError as e:
        print(f"✗ SciPy: {e}")
        return False
    
    try:
        import sklearn
        print("✓ scikit-learn")
    except ImportError as e:
        print(f"✗ scikit-learn: {e}")
        return False
    
    try:
        import matplotlib
        print("✓ Matplotlib")
    except ImportError as e:
        print(f"✗ Matplotlib: {e}")
        return False
    
    try:
        from smart_palletizier import (
            BoxDetector,
            PointCloudProcessor,
            PlanarPatchDetector,
            PoseEstimator,
            Visualizer
        )
        print("✓ smart_palletizier package")
    except ImportError as e:
        print(f"✗ smart_palletizier package: {e}")
        return False
    
    return True


def test_basic_functionality():
    """Test basic functionality of each module."""
    print("\nTesting basic functionality...")
    
    try:
        from smart_palletizier import BoxDetector
        detector = BoxDetector()
        print("✓ BoxDetector initialized")
    except Exception as e:
        print(f"✗ BoxDetector: {e}")
        return False
    
    try:
        from smart_palletizier import PointCloudProcessor
        processor = PointCloudProcessor()
        print("✓ PointCloudProcessor initialized")
    except Exception as e:
        print(f"✗ PointCloudProcessor: {e}")
        return False
    
    try:
        from smart_palletizier import PlanarPatchDetector
        planar_detector = PlanarPatchDetector()
        print("✓ PlanarPatchDetector initialized")
    except Exception as e:
        print(f"✗ PlanarPatchDetector: {e}")
        return False
    
    try:
        from smart_palletizier import PoseEstimator
        pose_estimator = PoseEstimator()
        print("✓ PoseEstimator initialized")
    except Exception as e:
        print(f"✗ PoseEstimator: {e}")
        return False
    
    try:
        from smart_palletizier import Visualizer
        visualizer = Visualizer()
        print("✓ Visualizer initialized")
    except Exception as e:
        print(f"✗ Visualizer: {e}")
        return False
    
    return True


def test_data_availability():
    """Check if data files are available."""
    print("\nChecking data availability...")
    from pathlib import Path
    
    data_dir = Path("data")
    
    if not data_dir.exists():
        print(f"✗ Data directory not found: {data_dir}")
        return False
    
    small_box = data_dir / "small_box"
    medium_box = data_dir / "medium_box"
    
    if small_box.exists():
        print(f"✓ Small box data found")
        
        # Check for specific files
        if (small_box / "color_image.png").exists():
            print("  ✓ color_image.png")
        if (small_box / "intrinsics.json").exists():
            print("  ✓ intrinsics.json")
        if list(small_box.glob("small_box_*_raw.ply")):
            print(f"  ✓ Point cloud files ({len(list(small_box.glob('small_box_*_raw.ply')))} files)")
    else:
        print(f"✗ Small box data not found")
    
    if medium_box.exists():
        print(f"✓ Medium box data found")
    else:
        print(f"✗ Medium box data not found")
    
    return True


def main():
    """Run all tests."""
    print("="*60)
    print("Smart Palletizer Installation Test")
    print("="*60)
    print()
    
    success = True
    
    # Test imports
    if not test_imports():
        success = False
    
    # Test functionality
    if not test_basic_functionality():
        success = False
    
    # Test data
    test_data_availability()  # Not critical for success
    
    print()
    print("="*60)
    if success:
        print("✓ All critical tests passed!")
        print("="*60)
        print()
        print("You can now run the pipeline:")
        print("  python -m smart_palletizier.pipeline --data-folder data/small_box --box-type small_box")
        return 0
    else:
        print("✗ Some tests failed. Please check the errors above.")
        print("="*60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
