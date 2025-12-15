"""
Point Cloud Processing Module

This module provides functionality for processing and cleaning point clouds,
including noise removal, downsampling, outlier removal, and statistical filtering.
"""

import open3d as o3d
import numpy as np
from typing import Tuple, Optional, List


class PointCloudProcessor:
    """
    A class for processing and cleaning point clouds.
    
    This processor implements various point cloud processing techniques including
    statistical outlier removal, radius outlier removal, voxel downsampling,
    and plane segmentation.
    
    Attributes:
        voxel_size (float): Voxel size for downsampling
        nb_neighbors (int): Number of neighbors for statistical outlier removal
        std_ratio (float): Standard deviation ratio for statistical outlier removal
    """
    
    def __init__(self, voxel_size: float = 0.002, nb_neighbors: int = 20,
                 std_ratio: float = 2.0):
        """
        Initialize the PointCloudProcessor.

        Args:
            voxel_size: Size of voxel for downsampling (in meters)
            nb_neighbors: Number of neighbors to analyze for outlier removal
            std_ratio: Standard deviation multiplier for outlier threshold
        """
        self.voxel_size = voxel_size
        self.nb_neighbors = nb_neighbors
        self.std_ratio = std_ratio
    
    def load_pointcloud(self, filepath: str) -> o3d.geometry.PointCloud:
        """
        Load a point cloud from file.
        
        Args:
            filepath: Path to the point cloud file (.ply, .pcd, etc.)
            
        Returns:
            Open3D PointCloud object
        """
        pcd = o3d.io.read_point_cloud(filepath)
        return pcd
    
    def save_pointcloud(self, pcd: o3d.geometry.PointCloud, filepath: str):
        """
        Save a point cloud to file.
        
        Args:
            pcd: Open3D PointCloud object
            filepath: Output file path
        """
        o3d.io.write_point_cloud(filepath, pcd)
    
    def remove_statistical_outliers(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """
        Remove statistical outliers from the point cloud.
        
        This method removes points that are further away from their neighbors
        compared to the average for the point cloud.
        
        Args:
            pcd: Input point cloud
            
        Returns:
            Filtered point cloud
        """
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=self.nb_neighbors,
                                                  std_ratio=self.std_ratio)
        return cl
    
    def remove_radius_outliers(self, pcd: o3d.geometry.PointCloud, 
                               radius: float = 0.01, min_points: int = 10) -> o3d.geometry.PointCloud:
        """
        Remove radius outliers from the point cloud.
        
        This method removes points that have few neighbors in a given sphere.
        
        Args:
            pcd: Input point cloud
            radius: Radius of the sphere for neighbor search (in meters)
            min_points: Minimum number of points required in the radius
            
        Returns:
            Filtered point cloud
        """
        cl, ind = pcd.remove_radius_outlier(nb_points=min_points, radius=radius)
        return cl
    
    def downsample(self, pcd: o3d.geometry.PointCloud, 
                   voxel_size: Optional[float] = None) -> o3d.geometry.PointCloud:
        """
        Downsample the point cloud using voxel grid filtering.
        
        Args:
            pcd: Input point cloud
            voxel_size: Size of voxel (if None, uses self.voxel_size)
            
        Returns:
            Downsampled point cloud
        """
        if voxel_size is None:
            voxel_size = self.voxel_size
        
        downsampled = pcd.voxel_down_sample(voxel_size=voxel_size)
        return downsampled
    
    def estimate_normals(self, pcd: o3d.geometry.PointCloud, 
                        radius: float = 0.01, max_nn: int = 30):
        """
        Estimate normals for the point cloud.
        
        Args:
            pcd: Input point cloud
            radius: Search radius for normal estimation
            max_nn: Maximum number of nearest neighbors
        """
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
        )
        # Orient normals consistently
        pcd.orient_normals_consistent_tangent_plane(k=15)
    
    def clean_pointcloud(self, pcd: o3d.geometry.PointCloud, 
                        downsample: bool = True,
                        remove_outliers: bool = True) -> o3d.geometry.PointCloud:
        """
        Apply a full cleaning pipeline to the point cloud.
        
        This method combines multiple filtering techniques to produce a clean
        point cloud suitable for further processing.
        
        Args:
            pcd: Input point cloud
            downsample: Whether to downsample the point cloud
            remove_outliers: Whether to remove outliers
            
        Returns:
            Cleaned point cloud
        """
        cleaned = pcd
        
        # Remove NaN and infinite values
        cleaned.remove_non_finite_points()
        
        if remove_outliers:
            # First pass: statistical outlier removal
            cleaned = self.remove_statistical_outliers(cleaned)
            
            # Second pass: radius outlier removal
            cleaned = self.remove_radius_outliers(cleaned, radius=0.01, min_points=5)
        
        if downsample:
            # Downsample to reduce density
            cleaned = self.downsample(cleaned)
        
        # Estimate normals for cleaned cloud
        if len(cleaned.points) > 0:
            self.estimate_normals(cleaned)
        
        return cleaned
    
    def segment_plane(self, pcd: o3d.geometry.PointCloud, 
                     distance_threshold: float = 0.01,
                     ransac_n: int = 3,
                     num_iterations: int = 1000) -> Tuple[np.ndarray, List[int]]:
        """
        Segment the largest plane from the point cloud using RANSAC.
        
        Args:
            pcd: Input point cloud
            distance_threshold: Maximum distance a point can be from the plane
            ransac_n: Number of points to sample for RANSAC
            num_iterations: Number of RANSAC iterations
            
        Returns:
            Tuple of (plane_model, inlier_indices)
            plane_model: [a, b, c, d] coefficients of plane equation ax+by+cz+d=0
        """
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations
        )
        return plane_model, inliers
    
    def extract_clusters(self, pcd: o3d.geometry.PointCloud,
                        eps: float = 0.02, min_points: int = 10) -> List[o3d.geometry.PointCloud]:
        """
        Extract clusters from the point cloud using DBSCAN.
        
        Args:
            pcd: Input point cloud
            eps: Maximum distance between two points to be in the same cluster
            min_points: Minimum number of points to form a cluster
            
        Returns:
            List of point cloud clusters
        """
        labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
        
        max_label = labels.max()
        clusters = []
        
        for i in range(max_label + 1):
            cluster_indices = np.where(labels == i)[0]
            cluster = pcd.select_by_index(cluster_indices)
            clusters.append(cluster)
        
        return clusters
    
    def compute_bounding_box(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.OrientedBoundingBox:
        """
        Compute the oriented bounding box of the point cloud.
        
        Args:
            pcd: Input point cloud
            
        Returns:
            Oriented bounding box
        """
        obb = pcd.get_oriented_bounding_box()
        obb.color = (1, 0, 0)  # Red color
        return obb
    
    def get_dimensions(self, pcd: o3d.geometry.PointCloud) -> np.ndarray:
        """
        Get the dimensions of the point cloud.
        
        Args:
            pcd: Input point cloud
            
        Returns:
            Array of [length, width, height] in meters
        """
        obb = self.compute_bounding_box(pcd)
        extent = obb.extent
        return extent
    
    def create_from_rgbd(self, color_image: np.ndarray, depth_image: np.ndarray,
                        intrinsics: dict, depth_scale: float = 1000.0,
                        depth_trunc: float = 3.0) -> o3d.geometry.PointCloud:
        """
        Create a point cloud from RGB-D images.
        
        Args:
            color_image: RGB color image (H x W x 3)
            depth_image: Depth image (H x W), values in millimeters
            intrinsics: Camera intrinsic parameters dict with 'fx', 'fy', 'cx', 'cy'
            depth_scale: Scale factor to convert depth to meters (default: 1000 for mm)
            depth_trunc: Maximum depth value to consider (in meters)
            
        Returns:
            Point cloud generated from RGBD
        """
        # Create Open3D images
        color_o3d = o3d.geometry.Image(color_image)
        depth_o3d = o3d.geometry.Image(depth_image)
        
        # Create RGBD image
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d, depth_o3d,
            depth_scale=depth_scale,
            depth_trunc=depth_trunc,
            convert_rgb_to_intensity=False
        )
        
        # Create camera intrinsic
        height, width = depth_image.shape
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width, height,
            intrinsics['fx'], intrinsics['fy'],
            intrinsics['cx'], intrinsics['cy']
        )
        
        # Generate point cloud
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
        
        return pcd
