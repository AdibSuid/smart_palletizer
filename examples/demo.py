"""
Example script demonstrating the Smart Palletizer pipeline usage.
"""

import cv2
import numpy as np
from pathlib import Path

from smart_palletizier.box_detector import BoxDetector
from smart_palletizier.pointcloud_processor import PointCloudProcessor
from smart_palletizier.planar_detector import PlanarPatchDetector
from smart_palletizier.pose_estimator import PoseEstimator
from smart_palletizier.visualization import Visualizer


def example_2d_detection():
    """Example: 2D box detection from image."""
    print("=== Example: 2D Box Detection ===\n")
    
    # Load image
    image_path = "data/small_box/color_image.png"
    image = cv2.imread(image_path)
    
    # Initialize detector
    detector = BoxDetector(min_area=1000, max_area=500000)
    
    # Detect boxes
    boxes = detector.detect(image)
    
    print(f"Detected {len(boxes)} boxes")
    for box in boxes:
        print(f"  Box {box['id']}: center={box['center']}, area={box['area']:.0f}")
    
    # Visualize
    result = detector.draw_detections(image, boxes)
    cv2.imshow("Detection Result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def example_pointcloud_processing():
    """Example: Point cloud cleaning."""
    print("\n=== Example: Point Cloud Processing ===\n")
    
    # Initialize processor
    processor = PointCloudProcessor(voxel_size=0.002, nb_neighbors=20, std_ratio=2.0)
    
    # Load point cloud
    pcd_path = "data/small_box/small_box_0_raw.ply"
    pcd = processor.load_pointcloud(pcd_path)
    
    print(f"Original points: {len(pcd.points)}")
    
    # Clean point cloud
    cleaned = processor.clean_pointcloud(pcd, downsample=True, remove_outliers=True)
    
    print(f"Cleaned points: {len(cleaned.points)}")
    
    # Get dimensions
    dims = processor.get_dimensions(cleaned)
    print(f"Dimensions: {dims}")
    
    # Visualize
    Visualizer.visualize_point_cloud(cleaned, "Cleaned Point Cloud")


def example_planar_detection():
    """Example: Planar patches detection."""
    print("\n=== Example: Planar Patches Detection ===\n")
    
    # Initialize processors
    pcd_processor = PointCloudProcessor()
    planar_detector = PlanarPatchDetector(distance_threshold=0.005, min_points=50)
    
    # Load and clean point cloud
    pcd = pcd_processor.load_pointcloud("data/small_box/small_box_0_raw.ply")
    cleaned = pcd_processor.clean_pointcloud(pcd)
    
    # Detect planes
    planes = planar_detector.detect_planes(cleaned, max_planes=10)
    
    print(f"Detected {len(planes)} planar patches")
    for plane in planes:
        print(f"  Plane {plane['id']}: {plane['num_points']} points, normal={plane['normal']}")
    
    # Analyze box structure
    if planes:
        analysis = planar_detector.analyze_box_from_planes(planes)
        print(f"\nBox Analysis:")
        print(f"  Center: {analysis['center']}")
        print(f"  Visible faces: {analysis['num_visible_faces']}")
        print(f"  Dimensions: {analysis['dimensions']}")
    
    # Visualize
    Visualizer.visualize_planar_patches(planes, cleaned)


def example_pose_estimation():
    """Example: 6D pose estimation."""
    print("\n=== Example: 6D Pose Estimation ===\n")
    
    # Initialize processors
    pcd_processor = PointCloudProcessor()
    pose_estimator = PoseEstimator()
    
    # Box dimensions (small box)
    box_dimensions = np.array([0.340, 0.250, 0.095])
    
    # Load and clean point cloud
    pcd = pcd_processor.load_pointcloud("data/small_box/small_box_0_raw.ply")
    cleaned = pcd_processor.clean_pointcloud(pcd)
    
    # Load template mesh
    template = pose_estimator.load_box_mesh("data/small_box/small_box_mesh.ply")
    
    # Estimate pose
    poses = pose_estimator.estimate_multiple_boxes([cleaned], template, box_dimensions)
    
    if poses:
        pose = poses[0]
        Visualizer.print_pose_info(pose)
        
        # Visualize
        Visualizer.visualize_poses([cleaned], poses, box_dimensions)


def example_complete_pipeline():
    """Example: Complete pipeline on one box."""
    print("\n=== Example: Complete Pipeline ===\n")
    
    data_folder = Path("data/small_box")
    
    # 1. 2D Detection
    print("Step 1: 2D Detection")
    image = cv2.imread(str(data_folder / "color_image.png"))
    detector = BoxDetector()
    
    # Use first mask
    mask = cv2.imread(str(data_folder / "small_box_mask_0.png"), cv2.IMREAD_GRAYSCALE)
    boxes_2d = detector.detect_boxes_from_mask(mask)
    print(f"  Detected {len(boxes_2d)} boxes in 2D")
    
    # 2. Point Cloud Processing
    print("\nStep 2: Point Cloud Processing")
    pcd_processor = PointCloudProcessor()
    pcd = pcd_processor.load_pointcloud(str(data_folder / "small_box_0_raw.ply"))
    cleaned = pcd_processor.clean_pointcloud(pcd)
    print(f"  Reduced from {len(pcd.points)} to {len(cleaned.points)} points")
    
    # 3. Planar Patches
    print("\nStep 3: Planar Patches Detection")
    planar_detector = PlanarPatchDetector()
    planes = planar_detector.detect_planes(cleaned, max_planes=10)
    print(f"  Detected {len(planes)} planar patches")
    
    # 4. Pose Estimation
    print("\nStep 4: Pose Estimation")
    pose_estimator = PoseEstimator()
    box_dimensions = np.array([0.340, 0.250, 0.095])
    
    try:
        template = pose_estimator.load_box_mesh(str(data_folder / "small_box_mesh.ply"))
        poses = pose_estimator.estimate_multiple_boxes([cleaned], template, box_dimensions)
        
        if poses:
            print(f"  Estimated pose for box")
            Visualizer.print_pose_info(poses[0])
    except FileNotFoundError:
        print("  Template mesh not found, using OBB-based estimation")
        poses = pose_estimator.estimate_multiple_boxes([cleaned], None, box_dimensions)
        if poses:
            Visualizer.print_pose_info(poses[0])
    
    print("\nPipeline completed!")


if __name__ == "__main__":
    # Run examples
    # Uncomment the ones you want to run
    
    # example_2d_detection()
    # example_pointcloud_processing()
    # example_planar_detection()
    # example_pose_estimation()
    example_complete_pipeline()
