"""
Box Extraction Module

Extracts clean individual boxes from point cloud scenes using clustering.
"""

import open3d as o3d
import numpy as np
from typing import List


class BoxExtractor:
    """
    Extracts individual boxes from point clouds that may contain background/noise.
    """

    def __init__(self, eps: float = 0.02, min_points: int = 50):
        """
        Initialize the BoxExtractor.

        Args:
            eps: DBSCAN clustering distance threshold
            min_points: Minimum points to form a cluster
        """
        self.eps = eps
        self.min_points = min_points

    def extract_largest_cluster(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """
        Extract the largest cluster from a point cloud.

        Args:
            pcd: Input point cloud

        Returns:
            Point cloud of the largest cluster
        """
        if len(pcd.points) < self.min_points:
            return pcd

        # Perform DBSCAN clustering
        labels = np.array(pcd.cluster_dbscan(
            eps=self.eps,
            min_points=self.min_points,
            print_progress=False
        ))

        if len(labels) == 0 or labels.max() < 0:
            return pcd

        # Find largest cluster
        unique_labels = np.unique(labels[labels >= 0])
        if len(unique_labels) == 0:
            return pcd

        cluster_sizes = [(label, np.sum(labels == label)) for label in unique_labels]
        largest_label = max(cluster_sizes, key=lambda x: x[1])[0]

        # Extract largest cluster
        indices = np.where(labels == largest_label)[0]
        largest_cluster = pcd.select_by_index(indices.tolist())

        return largest_cluster

    def extract_box_by_dimensions(self, pcd: o3d.geometry.PointCloud,
                                   expected_dims: np.ndarray,
                                   tolerance: float = 0.5) -> o3d.geometry.PointCloud:
        """
        Extract box region by expected dimensions.

        Args:
            pcd: Input point cloud
            expected_dims: Expected box dimensions [l, w, h]
            tolerance: Size tolerance multiplier

        Returns:
            Extracted box point cloud
        """
        points = np.asarray(pcd.points)

        if len(points) == 0:
            return pcd

        # Get center of mass
        center = points.mean(axis=0)

        # Define bounding box based on expected dimensions
        half_size = expected_dims * (1 + tolerance) / 2

        # Filter points within bounding box
        mask = np.all(np.abs(points - center) <= half_size, axis=1)
        indices = np.where(mask)[0]

        if len(indices) == 0:
            # Fallback: return largest cluster
            return self.extract_largest_cluster(pcd)

        extracted = pcd.select_by_index(indices.tolist())

        return extracted

    def remove_plane(self, pcd: o3d.geometry.PointCloud,
                     distance_threshold: float = 0.01,
                     ransac_n: int = 3,
                     num_iterations: int = 1000) -> o3d.geometry.PointCloud:
        """
        Remove the dominant plane (e.g., table surface) from point cloud.

        Args:
            pcd: Input point cloud
            distance_threshold: RANSAC distance threshold
            ransac_n: Number of points to sample
            num_iterations: RANSAC iterations

        Returns:
            Point cloud with plane removed
        """
        if len(pcd.points) < 10:
            return pcd

        # Segment largest plane
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations
        )

        # Remove plane points
        outliers = pcd.select_by_index(inliers, invert=True)

        return outliers

    def extract_box(self, pcd: o3d.geometry.PointCloud,
                    expected_dims: np.ndarray,
                    remove_table: bool = True) -> o3d.geometry.PointCloud:
        """
        Complete box extraction pipeline.

        Args:
            pcd: Input point cloud (may contain box + background)
            expected_dims: Expected box dimensions
            remove_table: Whether to remove table plane first

        Returns:
            Clean box point cloud
        """
        if len(pcd.points) < self.min_points:
            return pcd

        cleaned = pcd

        # Step 1: Remove table plane if requested
        if remove_table:
            # Check if there's a dominant horizontal plane
            cleaned = self.remove_plane(cleaned, distance_threshold=0.01)

        # Step 2: Cluster and get largest cluster
        if len(cleaned.points) > self.min_points:
            cleaned = self.extract_largest_cluster(cleaned)

        # Step 3: Crop to expected box dimensions
        if len(cleaned.points) > self.min_points:
            cleaned = self.extract_box_by_dimensions(cleaned, expected_dims, tolerance=0.3)

        return cleaned
