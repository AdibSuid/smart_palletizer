"""
Smart Palletizer - Main Pipeline

This script demonstrates the complete pipeline for:
1. 2D box detection
2. Point cloud processing
3. Planar patches detection
4. 6D pose estimation
"""

import cv2
import numpy as np
import open3d as o3d
import json
import argparse
import os
import warnings
from pathlib import Path

# Suppress warnings for headless environments
warnings.filterwarnings('ignore', category=UserWarning)
os.environ.setdefault('QT_QPA_PLATFORM', 'xcb')

from smart_palletizier.box_detector import BoxDetector
from smart_palletizier.pointcloud_processor import PointCloudProcessor
from smart_palletizier.planar_detector import PlanarPatchDetector
from smart_palletizier.pose_estimator import PoseEstimator
from smart_palletizier.visualization import Visualizer
from smart_palletizier.box_segmenter import BoxSegmenter
from smart_palletizier.box_extractor import BoxExtractor


def load_camera_intrinsics(filepath: str) -> dict:
    """Load camera intrinsic parameters from JSON file."""
    with open(filepath, 'r') as f:
        intrinsics = json.load(f)
    return intrinsics


def load_camera_transform(filepath: str) -> np.ndarray:
    """Load camera to root transformation from JSON file."""
    with open(filepath, 'r') as f:
        transform_data = json.load(f)
    
    # Convert to 4x4 matrix if needed
    if isinstance(transform_data, list):
        return np.array(transform_data)
    return transform_data


def run_task1_2d_detection(data_folder: Path, box_type: str = "small_box"):
    """
    Task 1: 2D Box Detection
    
    Detect boxes in color images using classical computer vision methods.
    """
    print(f"\n{'='*60}")
    print(f"Task 1: 2D Box Detection - {box_type}")
    print(f"{'='*60}")
    
    # Load images
    color_image_path = data_folder / "color_image.png"
    color_image = cv2.imread(str(color_image_path))
    
    # Initialize detector
    detector = BoxDetector(min_area=1000, max_area=500000)
    
    # Try to use mask if available
    mask_files = list(data_folder.glob(f"{box_type}_mask_*.png"))
    
    if mask_files:
        print(f"Found {len(mask_files)} mask files, using mask-based detection")
        all_boxes = []
        
        for mask_file in mask_files:
            mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
            boxes = detector.detect_boxes_from_mask(mask)
            all_boxes.extend(boxes)
        
        print(f"Detected {len(all_boxes)} boxes from masks")
    else:
        print("No masks found, using edge-based detection")
        all_boxes = detector.detect(color_image)
        print(f"Detected {len(all_boxes)} boxes")
    
    # Visualize results
    result_image = detector.draw_detections(color_image, all_boxes, box_type)
    
    # Save results
    output_path = data_folder / f"{box_type}_detection_result.png"
    cv2.imwrite(str(output_path), result_image)
    print(f"Results saved to: {output_path}")
    
    # Display (skip if headless)
    try:
        cv2.imshow(f"2D Detection - {box_type}", result_image)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()
    except:
        print("  Note: Display skipped (running in headless mode)")
    
    return all_boxes


def run_task2_planar_patches(data_folder: Path, box_type: str = "small_box"):
    """
    Task 2: Planar Patches Detection
    
    Detect planar surfaces in point clouds and group them by box.
    """
    print(f"\n{'='*60}")
    print(f"Task 2: Planar Patches Detection - {box_type}")
    print(f"{'='*60}")
    
    # Initialize processors
    pcd_processor = PointCloudProcessor()
    planar_detector = PlanarPatchDetector(distance_threshold=0.005, min_points=50)
    
    # Load point cloud files
    pcd_files = list(data_folder.glob(f"{box_type}_*_raw.ply"))
    
    if not pcd_files:
        print(f"No point cloud files found for {box_type}")
        return
    
    print(f"Processing {len(pcd_files)} point clouds")
    
    all_planes_by_box = []
    
    for i, pcd_file in enumerate(pcd_files[:3]):  # Process first 3 for demo
        print(f"\nProcessing: {pcd_file.name}")
        
        # Load and clean point cloud
        pcd = pcd_processor.load_pointcloud(str(pcd_file))
        print(f"  Original points: {len(pcd.points)}")
        
        # Clean point cloud
        cleaned_pcd = pcd_processor.clean_pointcloud(pcd, downsample=True, remove_outliers=True)
        print(f"  Cleaned points: {len(cleaned_pcd.points)}")
        
        # Detect planes
        planes = planar_detector.detect_planes(cleaned_pcd, max_planes=10)
        print(f"  Detected {len(planes)} planar patches")
        
        # Analyze box structure
        if planes:
            box_analysis = planar_detector.analyze_box_from_planes(planes)
            print(f"  Visible faces: {box_analysis['num_visible_faces']}")
            print(f"  Box center: {box_analysis['center']}")
            all_planes_by_box.append({'pcd': cleaned_pcd, 'planes': planes, 'analysis': box_analysis})
            
            # Visualize first box
            if i == 0:
                print("\n  Visualizing planar patches (close window to continue)...")
                try:
                    Visualizer.visualize_planar_patches(planes, cleaned_pcd, 
                                                       window_name=f"Planar Patches - {pcd_file.stem}")
                except Exception as e:
                    print(f"    Note: Visualization skipped (display not available): {e}")
    
    return all_planes_by_box


def run_task3_pointcloud_cleaning(data_folder: Path, box_type: str = "small_box"):
    """
    Task 3: Point Cloud Post-Processing
    
    Clean noisy point clouds while preserving box dimensions.
    """
    print(f"\n{'='*60}")
    print(f"Task 3: Point Cloud Post-Processing - {box_type}")
    print(f"{'='*60}")
    
    # Initialize processor
    pcd_processor = PointCloudProcessor(voxel_size=0.002, nb_neighbors=20, std_ratio=2.0)
    
    # Load first point cloud as example
    pcd_files = list(data_folder.glob(f"{box_type}_*_raw.ply"))
    
    if not pcd_files:
        print(f"No point cloud files found for {box_type}")
        return
    
    pcd_file = pcd_files[0]
    print(f"Processing: {pcd_file.name}")
    
    # Load raw point cloud
    raw_pcd = pcd_processor.load_pointcloud(str(pcd_file))
    print(f"Raw point cloud: {len(raw_pcd.points)} points")
    
    # Get dimensions before cleaning
    dims_before = pcd_processor.get_dimensions(raw_pcd)
    print(f"Dimensions before cleaning: {dims_before}")
    
    # Clean point cloud
    cleaned_pcd = pcd_processor.clean_pointcloud(raw_pcd, downsample=True, remove_outliers=True)
    print(f"Cleaned point cloud: {len(cleaned_pcd.points)} points")
    
    # Get dimensions after cleaning
    dims_after = pcd_processor.get_dimensions(cleaned_pcd)
    print(f"Dimensions after cleaning: {dims_after}")
    
    # Calculate dimension change
    dim_change = np.abs(dims_after - dims_before) / dims_before * 100
    print(f"Dimension change: {dim_change}%")
    
    # Save cleaned point cloud
    output_path = data_folder / f"{pcd_file.stem}_cleaned.ply"
    pcd_processor.save_pointcloud(cleaned_pcd, str(output_path))
    print(f"Cleaned point cloud saved to: {output_path}")
    
    # Visualize comparison
    print("\nVisualizing raw vs cleaned (close windows to continue)...")
    raw_pcd.paint_uniform_color([1, 0, 0])  # Red
    cleaned_pcd.paint_uniform_color([0, 1, 0])  # Green
    
    try:
        o3d.visualization.draw_geometries([raw_pcd], window_name="Raw Point Cloud")
        o3d.visualization.draw_geometries([cleaned_pcd], window_name="Cleaned Point Cloud")
    except Exception as e:
        print(f"  Note: Visualization skipped (display not available): {e}")
    
    return raw_pcd, cleaned_pcd


def run_task4_pose_estimation(data_folder: Path, box_type: str = "small_box"):
    """
    Task 4: 6D Pose Estimation

    Estimate position and orientation of boxes in the scene.
    """
    print(f"\n{'='*60}")
    print(f"Task 4: 6D Pose Estimation - {box_type}")
    print(f"{'='*60}")

    # Initialize processors
    pcd_processor = PointCloudProcessor()
    pose_estimator = PoseEstimator(icp_threshold=0.002, icp_max_iterations=2000)
    box_extractor = BoxExtractor(eps=0.02, min_points=50)

    # Define box dimensions (from task description)
    # Note: The mesh files were swapped - small_box_mesh.ply is actually medium size
    if box_type == "small_box":
        box_dimensions = np.array([0.255, 0.155, 0.100])  # meters (actually medium)
    else:  # medium_box
        box_dimensions = np.array([0.340, 0.250, 0.095])  # meters (actually small)

    print(f"Box dimensions: {box_dimensions}")

    # Load template mesh if available
    mesh_file = data_folder / f"{box_type}_mesh.ply"
    template_pcd = None

    if mesh_file.exists():
        print(f"Loading template mesh: {mesh_file.name}")
        template_pcd = pose_estimator.load_box_mesh(str(mesh_file))
        print(f"Template points: {len(template_pcd.points)}")

    # Load raw point cloud files (labeled per box but contain background)
    pcd_files = list(data_folder.glob(f"{box_type}_*_raw.ply"))
    print(f"\nProcessing {len(pcd_files)} point clouds")

    cleaned_clouds = []
    for i, pcd_file in enumerate(pcd_files):
        pcd = pcd_processor.load_pointcloud(str(pcd_file))

        # Simple cleaning - don't be too aggressive
        cleaned = pcd_processor.clean_pointcloud(pcd, downsample=True, remove_outliers=True)

        if len(cleaned.points) > 50:
            cleaned_clouds.append(cleaned)

    if not cleaned_clouds:
        print("No boxes found to process")
        return [], []

    # Estimate poses
    print(f"\nEstimating poses for {len(cleaned_clouds)} boxes...")
    poses = pose_estimator.estimate_multiple_boxes(cleaned_clouds, template_pcd, box_dimensions)

    print(f"\nEstimated poses for {len(poses)} boxes:")
    for pose in poses:
        Visualizer.print_pose_info(pose, pose['id'])

    # Save results
    results_file = data_folder / f"{box_type}_pose_results.txt"
    Visualizer.save_results_to_file(poses, str(results_file))
    print(f"Results saved to: {results_file}")

    # Visualize poses
    if poses:
        print("\nVisualizing estimated poses (close window to continue)...")
        try:
            Visualizer.visualize_poses(cleaned_clouds, poses, box_dimensions,
                                       window_name=f"Pose Estimation - {box_type}")
        except Exception as e:
            print(f"  Note: Visualization skipped (display not available): {e}")

    return poses, cleaned_clouds


def main():
    """Main pipeline execution."""
    parser = argparse.ArgumentParser(description="Smart Palletizer Pipeline")
    parser.add_argument("--data-folder", type=str, default="data/small_box",
                       help="Path to data folder (default: data/small_box)")
    parser.add_argument("--box-type", type=str, default="small_box",
                       choices=["small_box", "medium_box"],
                       help="Type of box to process")
    parser.add_argument("--tasks", type=str, default="all",
                       help="Tasks to run: 1,2,3,4 or 'all' (default: all)")
    
    args = parser.parse_args()
    
    data_folder = Path(args.data_folder)
    box_type = args.box_type
    
    if not data_folder.exists():
        print(f"Error: Data folder not found: {data_folder}")
        return
    
    print(f"\n{'#'*60}")
    print(f"# Smart Palletizer Pipeline")
    print(f"# Data folder: {data_folder}")
    print(f"# Box type: {box_type}")
    print(f"{'#'*60}")
    
    # Determine which tasks to run
    if args.tasks == "all":
        run_tasks = [1, 2, 3, 4]
    else:
        run_tasks = [int(t) for t in args.tasks.split(',')]
    
    # Run tasks
    if 1 in run_tasks:
        run_task1_2d_detection(data_folder, box_type)
    
    if 2 in run_tasks:
        run_task2_planar_patches(data_folder, box_type)
    
    if 3 in run_tasks:
        run_task3_pointcloud_cleaning(data_folder, box_type)
    
    if 4 in run_tasks:
        run_task4_pose_estimation(data_folder, box_type)
    
    print(f"\n{'#'*60}")
    print("# Pipeline completed!")
    print(f"{'#'*60}\n")


if __name__ == "__main__":
    main()
