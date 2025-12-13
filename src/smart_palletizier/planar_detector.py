"""
Planar Patches Detection Module

This module provides functionality for detecting and grouping planar surfaces
in point clouds, particularly useful for identifying box faces and their orientations.
"""

import open3d as o3d
import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.cluster import DBSCAN


class PlanarPatchDetector:
    """
    A class for detecting planar patches in point clouds.
    
    This detector uses RANSAC-based plane segmentation to identify planar surfaces
    and groups them according to their spatial proximity and normal orientation.
    
    Attributes:
        distance_threshold (float): Max distance for plane inlier points
        ransac_n (int): Number of points for RANSAC sampling
        num_iterations (int): Number of RANSAC iterations
        min_points (int): Minimum points required for a valid plane
    """
    
    def __init__(self, distance_threshold: float = 0.005,
                 ransac_n: int = 3,
                 num_iterations: int = 1000,
                 min_points: int = 50):
        """
        Initialize the PlanarPatchDetector.
        
        Args:
            distance_threshold: Maximum distance a point can be from plane model
            ransac_n: Number of points to randomly sample for RANSAC
            num_iterations: Number of iterations for RANSAC
            min_points: Minimum number of points to form a valid plane
        """
        self.distance_threshold = distance_threshold
        self.ransac_n = ransac_n
        self.num_iterations = num_iterations
        self.min_points = min_points
    
    def detect_planes(self, pcd: o3d.geometry.PointCloud, 
                     max_planes: int = 10) -> List[Dict]:
        """
        Detect multiple planar surfaces in the point cloud.
        
        Uses iterative RANSAC to detect multiple planes by removing
        inliers after each detection.
        
        Args:
            pcd: Input point cloud
            max_planes: Maximum number of planes to detect
            
        Returns:
            List of dictionaries containing plane information
        """
        planes = []
        remaining_pcd = pcd
        
        for i in range(max_planes):
            if len(remaining_pcd.points) < self.min_points:
                break
            
            # Segment plane
            plane_model, inliers = remaining_pcd.segment_plane(
                distance_threshold=self.distance_threshold,
                ransac_n=self.ransac_n,
                num_iterations=self.num_iterations
            )
            
            if len(inliers) < self.min_points:
                break
            
            # Extract plane points
            plane_cloud = remaining_pcd.select_by_index(inliers)
            
            # Calculate plane properties
            a, b, c, d = plane_model
            normal = np.array([a, b, c])
            normal = normal / np.linalg.norm(normal)  # Normalize
            
            # Calculate plane center
            points = np.asarray(plane_cloud.points)
            center = np.mean(points, axis=0)
            
            # Calculate plane area (approximate)
            area = len(inliers) * (self.distance_threshold ** 2)
            
            # Compute bounding box for the plane
            obb = plane_cloud.get_oriented_bounding_box()
            
            planes.append({
                'id': i,
                'model': plane_model,
                'normal': normal,
                'center': center,
                'inliers': inliers,
                'cloud': plane_cloud,
                'area': area,
                'num_points': len(inliers),
                'obb': obb
            })
            
            # Remove inliers from remaining points
            remaining_pcd = remaining_pcd.select_by_index(inliers, invert=True)
        
        return planes
    
    def group_planes_by_orientation(self, planes: List[Dict], 
                                    angle_threshold: float = 10.0) -> Dict[str, List[Dict]]:
        """
        Group planes by their normal orientation (X, Y, Z aligned).
        
        Args:
            planes: List of plane dictionaries from detect_planes
            angle_threshold: Maximum angle deviation in degrees
            
        Returns:
            Dictionary mapping orientation to list of planes
        """
        # Define principal axes
        axes = {
            'X': np.array([1, 0, 0]),
            'Y': np.array([0, 1, 0]),
            'Z': np.array([0, 0, 1])
        }
        
        grouped = {'X': [], 'Y': [], 'Z': [], 'other': []}
        angle_thresh_rad = np.radians(angle_threshold)
        
        for plane in planes:
            normal = plane['normal']
            
            # Find closest axis
            best_axis = None
            min_angle = float('inf')
            
            for axis_name, axis_vec in axes.items():
                # Calculate angle between normal and axis
                dot_product = np.abs(np.dot(normal, axis_vec))
                dot_product = np.clip(dot_product, -1.0, 1.0)
                angle = np.arccos(dot_product)
                
                if angle < min_angle:
                    min_angle = angle
                    best_axis = axis_name
            
            # Group by best matching axis if within threshold
            if min_angle < angle_thresh_rad:
                grouped[best_axis].append(plane)
            else:
                grouped['other'].append(plane)
        
        return grouped
    
    def group_planes_by_proximity(self, planes: List[Dict],
                                  distance_threshold: float = 0.1) -> List[List[Dict]]:
        """
        Group planes by spatial proximity (likely belonging to same box).
        
        Args:
            planes: List of plane dictionaries
            distance_threshold: Maximum distance between plane centers
            
        Returns:
            List of plane groups (each group is a list of planes)
        """
        if len(planes) == 0:
            return []
        
        # Extract plane centers
        centers = np.array([plane['center'] for plane in planes])
        
        # Cluster using DBSCAN
        clustering = DBSCAN(eps=distance_threshold, min_samples=1)
        labels = clustering.fit_predict(centers)
        
        # Group planes by cluster label
        groups = []
        for label in np.unique(labels):
            if label == -1:  # Noise points
                continue
            
            group_indices = np.where(labels == label)[0]
            group = [planes[i] for i in group_indices]
            groups.append(group)
        
        return groups
    
    def classify_box_faces(self, plane_group: List[Dict]) -> Dict[str, Optional[Dict]]:
        """
        Classify planes in a group as box faces (top, bottom, front, back, left, right).
        
        Args:
            plane_group: List of planes belonging to the same box
            
        Returns:
            Dictionary mapping face name to plane
        """
        # Group by orientation
        oriented = self.group_planes_by_orientation(plane_group, angle_threshold=15.0)
        
        faces = {
            'top': None,
            'bottom': None,
            'front': None,
            'back': None,
            'left': None,
            'right': None
        }
        
        # Z-axis aligned planes (top/bottom)
        if len(oriented['Z']) > 0:
            z_planes = sorted(oriented['Z'], key=lambda p: p['center'][2], reverse=True)
            faces['top'] = z_planes[0]
            if len(z_planes) > 1:
                faces['bottom'] = z_planes[-1]
        
        # Y-axis aligned planes (front/back)
        if len(oriented['Y']) > 0:
            y_planes = sorted(oriented['Y'], key=lambda p: p['center'][1])
            faces['front'] = y_planes[0]
            if len(y_planes) > 1:
                faces['back'] = y_planes[-1]
        
        # X-axis aligned planes (left/right)
        if len(oriented['X']) > 0:
            x_planes = sorted(oriented['X'], key=lambda p: p['center'][0])
            faces['left'] = x_planes[0]
            if len(x_planes) > 1:
                faces['right'] = x_planes[-1]
        
        return faces
    
    def visualize_planes(self, planes: List[Dict], 
                        original_pcd: Optional[o3d.geometry.PointCloud] = None) -> List[o3d.geometry.TriangleMesh]:
        """
        Create visualization meshes for detected planes.
        
        Args:
            planes: List of detected planes
            original_pcd: Optional original point cloud for context
            
        Returns:
            List of coordinate frames for each plane
        """
        geometries = []
        
        # Add original point cloud if provided
        if original_pcd is not None:
            geometries.append(original_pcd)
        
        # Color map for different planes
        colors = [
            [1, 0, 0],  # Red
            [0, 1, 0],  # Green
            [0, 0, 1],  # Blue
            [1, 1, 0],  # Yellow
            [1, 0, 1],  # Magenta
            [0, 1, 1],  # Cyan
        ]
        
        for i, plane in enumerate(planes):
            # Color the plane cloud
            plane_cloud = plane['cloud']
            color = colors[i % len(colors)]
            plane_cloud.paint_uniform_color(color)
            geometries.append(plane_cloud)
            
            # Add coordinate frame at plane center
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=0.05, origin=plane['center']
            )
            geometries.append(frame)
            
            # Add oriented bounding box
            if 'obb' in plane:
                obb = plane['obb']
                obb.color = color
                geometries.append(obb)
        
        return geometries
    
    def compute_plane_dimensions(self, plane: Dict) -> Tuple[float, float]:
        """
        Compute the dimensions (width, height) of a planar patch.
        
        Args:
            plane: Plane dictionary from detect_planes
            
        Returns:
            Tuple of (width, height) in meters
        """
        obb = plane['obb']
        extent = obb.extent
        
        # Get the two largest dimensions (ignoring thickness)
        sorted_extent = np.sort(extent)
        width = sorted_extent[1]
        height = sorted_extent[2]
        
        return width, height
    
    def analyze_box_from_planes(self, plane_group: List[Dict]) -> Dict:
        """
        Analyze a box structure from its planar patches.
        
        Args:
            plane_group: List of planes belonging to a single box
            
        Returns:
            Dictionary with box analysis results
        """
        faces = self.classify_box_faces(plane_group)
        
        # Compute box center from all planes
        all_centers = np.array([p['center'] for p in plane_group])
        box_center = np.mean(all_centers, axis=0)
        
        # Estimate box dimensions from faces
        dimensions = {'length': 0, 'width': 0, 'height': 0}
        
        if faces['top'] is not None or faces['bottom'] is not None:
            face = faces['top'] if faces['top'] is not None else faces['bottom']
            w, h = self.compute_plane_dimensions(face)
            dimensions['length'] = max(w, h)
            dimensions['width'] = min(w, h)
        
        if faces['front'] is not None or faces['back'] is not None:
            face = faces['front'] if faces['front'] is not None else faces['back']
            w, h = self.compute_plane_dimensions(face)
            dimensions['height'] = h
        
        return {
            'center': box_center,
            'dimensions': dimensions,
            'faces': faces,
            'num_visible_faces': sum(1 for f in faces.values() if f is not None),
            'planes': plane_group
        }
