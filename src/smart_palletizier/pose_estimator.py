"""
Pose Estimation Module

This module provides functionality for estimating 6D poses (position and orientation)
of boxes using point cloud registration techniques like ICP and feature-based matching.
"""

import open3d as o3d
import numpy as np
from typing import Dict, Tuple, Optional, List
from scipy.spatial.transform import Rotation as R


class PoseEstimator:
    """
    A class for estimating 6D poses of objects from point clouds.
    
    This estimator uses various registration techniques including ICP (Iterative Closest Point),
    PCA-based alignment, and template matching to estimate object poses.
    
    Attributes:
        icp_threshold (float): Distance threshold for ICP convergence
        icp_max_iterations (int): Maximum iterations for ICP
    """
    
    def __init__(self, icp_threshold: float = 0.01, icp_max_iterations: int = 2000):
        """
        Initialize the PoseEstimator.

        Args:
            icp_threshold: Distance threshold for ICP (default 0.01m = 1cm)
            icp_max_iterations: Maximum number of ICP iterations
        """
        self.icp_threshold = icp_threshold
        self.icp_max_iterations = icp_max_iterations
    
    def load_box_mesh(self, filepath: str) -> o3d.geometry.PointCloud:
        """
        Load a box mesh and convert to point cloud template.

        Args:
            filepath: Path to the mesh file

        Returns:
            Point cloud sampled from the mesh
        """
        mesh = o3d.io.read_triangle_mesh(filepath)

        # Check if mesh is in millimeters and convert to meters
        vertices = np.asarray(mesh.vertices)
        max_dim = np.max(vertices.max(axis=0) - vertices.min(axis=0))

        # If largest dimension > 10, assume it's in millimeters
        if max_dim > 10:
            mesh.scale(0.001, center=(0, 0, 0))

        # Sample points from mesh
        pcd = mesh.sample_points_uniformly(number_of_points=5000)
        return pcd
    
    def estimate_pose_icp(self, source: o3d.geometry.PointCloud,
                          target: o3d.geometry.PointCloud,
                          initial_transform: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict]:
        """
        Estimate pose using Iterative Closest Point (ICP) registration.
        
        Args:
            source: Source point cloud (template/model)
            target: Target point cloud (scene/observed)
            initial_transform: Initial 4x4 transformation matrix
            
        Returns:
            Tuple of (transformation_matrix, registration_result_dict)
        """
        if initial_transform is None:
            initial_transform = np.eye(4)
        
        # Make sure both clouds have normals for point-to-plane ICP
        # Point-to-plane generally gives better results than point-to-point
        if not source.has_normals():
            source.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30)
            )
        
        if not target.has_normals():
            target.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30)
            )
        
        # Point-to-plane ICP - converges faster and more accurately
        reg_p2p = o3d.pipelines.registration.registration_icp(
            source, target, self.icp_threshold, initial_transform,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=self.icp_max_iterations
            )
        )
        
        result = {
            'fitness': reg_p2p.fitness,
            'inlier_rmse': reg_p2p.inlier_rmse,
            'transformation': reg_p2p.transformation
        }
        
        return reg_p2p.transformation, result
    
    def estimate_initial_pose_pca(self, pcd: o3d.geometry.PointCloud,
                                   box_dimensions: np.ndarray) -> np.ndarray:
        """
        Estimate initial pose using PCA-based alignment.
        
        Args:
            pcd: Input point cloud
            box_dimensions: Expected box dimensions [length, width, height]
            
        Returns:
            4x4 transformation matrix
        """
        # Compute center
        points = np.asarray(pcd.points)
        center = np.mean(points, axis=0)
        
        # Center the points
        centered_points = points - center
        
        # Compute PCA
        covariance = np.cov(centered_points.T)
        eigenvalues, eigenvectors = np.linalg.eig(covariance)
        
        # Sort eigenvectors by eigenvalues
        idx = eigenvalues.argsort()[::-1]
        eigenvectors = eigenvectors[:, idx]
        
        # Ensure right-handed coordinate system
        if np.linalg.det(eigenvectors) < 0:
            eigenvectors[:, 2] = -eigenvectors[:, 2]
        
        # Create transformation matrix
        transform = np.eye(4)
        transform[:3, :3] = eigenvectors
        transform[:3, 3] = center
        
        return transform
    
    def estimate_pose_from_obb(self, obb: o3d.geometry.OrientedBoundingBox) -> Dict:
        """
        Extract pose from an oriented bounding box.
        
        Args:
            obb: Oriented bounding box
            
        Returns:
            Dictionary with position, rotation matrix, and euler angles
        """
        center = obb.center
        rotation = obb.R
        extent = obb.extent
        
        # Create transformation matrix
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = center
        
        # Convert rotation to euler angles (XYZ convention)
        rot = R.from_matrix(rotation)
        euler_angles = rot.as_euler('xyz', degrees=True)
        quaternion = rot.as_quat()  # [x, y, z, w]
        
        return {
            'position': center,
            'rotation_matrix': rotation,
            'euler_angles': euler_angles,
            'quaternion': quaternion,
            'transformation': transform,
            'dimensions': extent
        }
    
    def refine_pose_with_template(self, observed_pcd: o3d.geometry.PointCloud,
                                   template_pcd: o3d.geometry.PointCloud,
                                   initial_pose: np.ndarray) -> Tuple[Dict, o3d.geometry.PointCloud]:
        """
        Refine pose estimation using multi-scale template matching with ICP.

        Args:
            observed_pcd: Observed point cloud from scene
            template_pcd: Template point cloud (box model)
            initial_pose: Initial 4x4 transformation estimate

        Returns:
            Tuple of (pose_dict, aligned_template)
        """
        # Apply initial transformation to template
        template_copy = template_pcd.__copy__()

        # Multi-scale ICP: coarse to fine
        # Stage 1: Coarse alignment with higher threshold
        coarse_threshold = self.icp_threshold * 5  # 5x more tolerant
        template_stage1 = template_copy.__copy__()
        template_stage1.transform(initial_pose)

        transformation_1, _ = self.estimate_pose_icp(
            template_stage1, observed_pcd, np.eye(4)
        )
        intermediate_transform = transformation_1 @ initial_pose

        # Stage 2: Medium alignment
        medium_threshold = self.icp_threshold * 2
        template_stage2 = template_copy.__copy__()
        template_stage2.transform(intermediate_transform)

        # Temporarily update threshold
        original_threshold = self.icp_threshold
        self.icp_threshold = medium_threshold
        transformation_2, _ = self.estimate_pose_icp(
            template_stage2, observed_pcd, np.eye(4)
        )
        refined_transform = transformation_2 @ intermediate_transform

        # Stage 3: Fine alignment with original threshold
        self.icp_threshold = original_threshold
        template_stage3 = template_copy.__copy__()
        template_stage3.transform(refined_transform)

        transformation_3, reg_result = self.estimate_pose_icp(
            template_stage3, observed_pcd, np.eye(4)
        )

        # Combine all transformations
        final_transform = transformation_3 @ refined_transform

        # Apply final transformation
        template_copy.transform(final_transform)

        # Extract pose components
        rotation = final_transform[:3, :3]
        translation = final_transform[:3, 3]

        rot = R.from_matrix(rotation)
        euler_angles = rot.as_euler('xyz', degrees=True)
        quaternion = rot.as_quat()

        pose = {
            'position': translation,
            'rotation_matrix': rotation,
            'euler_angles': euler_angles,
            'quaternion': quaternion,
            'transformation': final_transform,
            'fitness': reg_result['fitness'],
            'inlier_rmse': reg_result['inlier_rmse']
        }

        return pose, template_copy
    
    def estimate_multiple_boxes(self, point_clouds: List[o3d.geometry.PointCloud],
                                template_pcd: o3d.geometry.PointCloud,
                                box_dimensions: np.ndarray,
                                min_fitness: float = 0.01) -> List[Dict]:
        """
        Estimate poses for multiple boxes.

        Args:
            point_clouds: List of segmented box point clouds
            template_pcd: Template point cloud for matching
            box_dimensions: Expected box dimensions
            min_fitness: Minimum ICP fitness to accept (default 1%)

        Returns:
            List of pose dictionaries (only successful poses)
        """
        poses = []

        for i, pcd in enumerate(point_clouds):
            if len(pcd.points) < 10:
                continue

            # Get OBB for initial pose
            obb = pcd.get_oriented_bounding_box()
            initial_pose_dict = self.estimate_pose_from_obb(obb)

            # Try to refine with ICP if template is provided
            if template_pcd is not None and len(pcd.points) > 50:
                try:
                    refined_pose, aligned = self.refine_pose_with_template(
                        pcd, template_pcd, initial_pose_dict['transformation']
                    )
                    refined_pose['id'] = i
                    refined_pose['method'] = 'ICP_refined'

                    # Only add pose if ICP fitness is good enough
                    if refined_pose.get('fitness', 0) >= min_fitness:
                        poses.append(refined_pose)
                    else:
                        print(f"    ⚠ Box {i}: ICP fitness {refined_pose.get('fitness', 0):.1%} too low - skipped")
                except:
                    # Fall back to OBB-based pose
                    initial_pose_dict['id'] = i
                    initial_pose_dict['method'] = 'OBB_only'
                    initial_pose_dict['fitness'] = 0.0
                    # Don't add OBB-only poses as they're unreliable
                    print(f"    ⚠ Box {i}: ICP failed - skipped")
            else:
                # Don't add boxes without ICP refinement
                print(f"    ⚠ Box {i}: Too few points or no template - skipped")

        return poses
    
    def create_box_mesh(self, dimensions: np.ndarray, pose: Dict) -> o3d.geometry.TriangleMesh:
        """
        Create a box mesh at the estimated pose.
        
        Args:
            dimensions: Box dimensions [length, width, height]
            pose: Pose dictionary with transformation
            
        Returns:
            Transformed box mesh
        """
        # Create box mesh centered at origin
        box = o3d.geometry.TriangleMesh.create_box(
            width=dimensions[0],
            height=dimensions[1],
            depth=dimensions[2]
        )
        
        # Translate to center the box
        box.translate(-dimensions / 2)
        
        # Apply pose transformation
        box.transform(pose['transformation'])
        
        # Color the box
        box.paint_uniform_color([0.7, 0.7, 0.7])
        box.compute_vertex_normals()
        
        return box
    
    def create_coordinate_frame(self, pose: Dict, size: float = 0.1) -> o3d.geometry.TriangleMesh:
        """
        Create a coordinate frame at the estimated pose.
        
        Args:
            pose: Pose dictionary with transformation
            size: Size of the coordinate frame axes
            
        Returns:
            Coordinate frame mesh
        """
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
        frame.transform(pose['transformation'])
        return frame
    
    def compute_pose_error(self, estimated_pose: Dict, 
                          ground_truth_pose: Dict) -> Dict[str, float]:
        """
        Compute error metrics between estimated and ground truth poses.
        
        Args:
            estimated_pose: Estimated pose dictionary
            ground_truth_pose: Ground truth pose dictionary
            
        Returns:
            Dictionary with error metrics
        """
        # Position error (Euclidean distance)
        pos_error = np.linalg.norm(
            estimated_pose['position'] - ground_truth_pose['position']
        )
        
        # Rotation error (angle between rotation matrices)
        R_est = estimated_pose['rotation_matrix']
        R_gt = ground_truth_pose['rotation_matrix']
        R_diff = R_gt.T @ R_est
        
        # Angle from rotation matrix
        trace = np.trace(R_diff)
        angle_error = np.arccos((trace - 1) / 2)
        angle_error_deg = np.degrees(angle_error)
        
        return {
            'position_error_m': pos_error,
            'rotation_error_deg': angle_error_deg,
            'rotation_error_rad': angle_error
        }
    
    def pose_to_dict(self, transformation: np.ndarray) -> Dict:
        """
        Convert a 4x4 transformation matrix to a pose dictionary.
        
        Args:
            transformation: 4x4 transformation matrix
            
        Returns:
            Pose dictionary
        """
        rotation = transformation[:3, :3]
        translation = transformation[:3, 3]
        
        rot = R.from_matrix(rotation)
        euler_angles = rot.as_euler('xyz', degrees=True)
        quaternion = rot.as_quat()
        
        return {
            'position': translation,
            'rotation_matrix': rotation,
            'euler_angles': euler_angles,
            'quaternion': quaternion,
            'transformation': transformation
        }
