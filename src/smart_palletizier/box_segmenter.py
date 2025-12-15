"""
Box Segmentation Module

This module segments individual boxes from point cloud scenes by combining
2D detection results with 3D point cloud data.
"""

import cv2
import numpy as np
import open3d as o3d
from typing import List, Tuple, Dict
import json


class BoxSegmenter:
    """
    Segments individual boxes from a point cloud using 2D detection and depth.
    """

    def __init__(self):
        """Initialize the BoxSegmenter."""
        pass

    def load_camera_intrinsics(self, filepath: str) -> o3d.camera.PinholeCameraIntrinsic:
        """
        Load camera intrinsic parameters.

        Args:
            filepath: Path to camera intrinsics JSON file

        Returns:
            Open3D PinholeCameraIntrinsic object
        """
        with open(filepath, 'r') as f:
            intrinsics = json.load(f)

        width = intrinsics.get('width', 640)
        height = intrinsics.get('height', 480)
        fx = intrinsics['fx']
        fy = intrinsics['fy']
        cx = intrinsics['cx']
        cy = intrinsics['cy']

        return o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

    def create_pointcloud_from_rgbd(self, color_image: np.ndarray,
                                     depth_image: np.ndarray,
                                     intrinsics: o3d.camera.PinholeCameraIntrinsic) -> o3d.geometry.PointCloud:
        """
        Create point cloud from RGB-D images.

        Args:
            color_image: RGB image
            depth_image: Depth image (in millimeters)
            intrinsics: Camera intrinsic parameters

        Returns:
            Point cloud
        """
        # Convert to Open3D images
        color_o3d = o3d.geometry.Image(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
        depth_o3d = o3d.geometry.Image(depth_image.astype(np.uint16))

        # Create RGBD image
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d, depth_o3d,
            depth_scale=1000.0,  # mm to meters
            depth_trunc=3.0,
            convert_rgb_to_intensity=False
        )

        # Create point cloud
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsics)

        return pcd

    def segment_box_from_mask(self, pointcloud: o3d.geometry.PointCloud,
                               mask: np.ndarray,
                               intrinsics: o3d.camera.PinholeCameraIntrinsic,
                               margin: int = 5) -> o3d.geometry.PointCloud:
        """
        Segment a box from point cloud using a 2D mask.

        Args:
            pointcloud: Full scene point cloud
            mask: Binary mask of the box (2D image)
            intrinsics: Camera intrinsics
            margin: Pixel margin to expand mask

        Returns:
            Segmented box point cloud
        """
        # Get points and colors
        points = np.asarray(pointcloud.points)

        if len(points) == 0:
            return o3d.geometry.PointCloud()

        # Project 3D points to 2D image plane
        fx = intrinsics.intrinsic_matrix[0, 0]
        fy = intrinsics.intrinsic_matrix[1, 1]
        cx = intrinsics.intrinsic_matrix[0, 2]
        cy = intrinsics.intrinsic_matrix[1, 2]

        # Project points
        x_2d = (points[:, 0] * fx / points[:, 2] + cx).astype(int)
        y_2d = (points[:, 1] * fy / points[:, 2] + cy).astype(int)

        # Expand mask slightly
        kernel = np.ones((margin*2+1, margin*2+1), np.uint8)
        mask_expanded = cv2.dilate(mask, kernel, iterations=1)

        # Filter points within mask
        h, w = mask_expanded.shape
        valid_idx = []

        for i in range(len(points)):
            if 0 <= x_2d[i] < w and 0 <= y_2d[i] < h:
                if mask_expanded[y_2d[i], x_2d[i]] > 0 and points[i, 2] > 0:
                    valid_idx.append(i)

        if len(valid_idx) == 0:
            return o3d.geometry.PointCloud()

        # Create segmented point cloud
        segmented_pcd = pointcloud.select_by_index(valid_idx)

        return segmented_pcd

    def segment_boxes_from_scene(self, color_image: np.ndarray,
                                  depth_image: np.ndarray,
                                  masks: List[np.ndarray],
                                  intrinsics: o3d.camera.PinholeCameraIntrinsic) -> List[o3d.geometry.PointCloud]:
        """
        Segment multiple boxes from a scene.

        Args:
            color_image: RGB image
            depth_image: Depth image
            masks: List of binary masks for each box
            intrinsics: Camera intrinsics

        Returns:
            List of segmented box point clouds
        """
        # Create full scene point cloud
        full_pcd = self.create_pointcloud_from_rgbd(color_image, depth_image, intrinsics)

        # Segment each box
        box_clouds = []
        for mask in masks:
            box_pcd = self.segment_box_from_mask(full_pcd, mask, intrinsics)

            # Filter by minimum number of points
            if len(box_pcd.points) > 50:
                box_clouds.append(box_pcd)

        return box_clouds

    def crop_box_region(self, pointcloud: o3d.geometry.PointCloud,
                        box_dimensions: np.ndarray,
                        tolerance: float = 1.5) -> o3d.geometry.PointCloud:
        """
        Crop point cloud to expected box region with tolerance.

        Args:
            pointcloud: Input point cloud
            box_dimensions: Expected box dimensions [l, w, h]
            tolerance: Multiplier for cropping region

        Returns:
            Cropped point cloud
        """
        points = np.asarray(pointcloud.points)

        if len(points) == 0:
            return pointcloud

        # Get center and extent
        center = points.mean(axis=0)
        extent = points.max(axis=0) - points.min(axis=0)

        # Define crop box
        max_expected_dim = np.max(box_dimensions) * tolerance

        # Filter points within reasonable distance from center
        distances_from_center = np.linalg.norm(points - center, axis=1)
        valid_idx = np.where(distances_from_center < max_expected_dim)[0]

        if len(valid_idx) > 0:
            cropped = pointcloud.select_by_index(valid_idx.tolist())
            return cropped

        return pointcloud
