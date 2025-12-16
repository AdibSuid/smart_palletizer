"""
Visualization Utilities

This module provides visualization functions for displaying detection results,
point clouds, and pose estimations.
"""

import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import json
from typing import List, Dict, Optional, Tuple
from pathlib import Path


class Visualizer:
    """
    A class for visualizing point clouds, detections, and poses.
    
    Provides functions for creating visualizations of 2D detections,
    3D point clouds, planar patches, and estimated poses.
    """
    
    @staticmethod
    def visualize_2d_detections(image: np.ndarray, boxes: List[Dict],
                                box_type: str = "box",
                                save_path: Optional[str] = None) -> np.ndarray:
        """
        Visualize 2D box detections on an image.
        
        Args:
            image: Input image
            boxes: List of detected boxes
            box_type: Label for the box type
            save_path: Optional path to save the visualization
            
        Returns:
            Image with visualized detections
        """
        result = image.copy()
        
        for box in boxes:
            # Draw bounding box
            x, y, w, h = box['bbox']
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw rotated rectangle
            if 'box_points' in box:
                cv2.drawContours(result, [box['box_points']], 0, (255, 0, 0), 2)
            
            # Draw center
            cx, cy = box['center']
            cv2.circle(result, (cx, cy), 5, (0, 0, 255), -1)
            
            # Add label
            label = f"{box_type}_{box['id']}"
            cv2.putText(result, label, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        if save_path:
            cv2.imwrite(save_path, result)
        
        return result
    
    @staticmethod
    def visualize_point_cloud(pcd: o3d.geometry.PointCloud,
                             window_name: str = "Point Cloud",
                             show_normals: bool = False):
        """
        Visualize a single point cloud.
        
        Args:
            pcd: Point cloud to visualize
            window_name: Name of the visualization window
            show_normals: Whether to show normals
        """
        geometries = [pcd]
        
        if show_normals and pcd.has_normals():
            # Can't directly show normals, but we can indicate it
            pass
        
        o3d.visualization.draw_geometries(
            geometries,
            window_name=window_name,
            width=1024,
            height=768
        )
    
    @staticmethod
    def visualize_multiple_geometries(geometries: List,
                                      window_name: str = "Visualization"):
        """
        Visualize multiple geometries together.
        
        Args:
            geometries: List of Open3D geometries
            window_name: Name of the visualization window
        """
        o3d.visualization.draw_geometries(
            geometries,
            window_name=window_name,
            width=1280,
            height=960
        )
    
    @staticmethod
    def visualize_planar_patches(planes: List[Dict],
                                 original_pcd: Optional[o3d.geometry.PointCloud] = None,
                                 window_name: str = "Planar Patches"):
        """
        Visualize detected planar patches.
        
        Args:
            planes: List of detected plane dictionaries
            original_pcd: Optional original point cloud
            window_name: Window title
        """
        geometries = []
        
        # Add original point cloud if provided
        if original_pcd is not None:
            original_copy = original_pcd.__copy__()
            original_copy.paint_uniform_color([0.5, 0.5, 0.5])
            geometries.append(original_copy)
        
        # Color palette for planes
        colors = [
            [1, 0, 0],    # Red
            [0, 1, 0],    # Green
            [0, 0, 1],    # Blue
            [1, 1, 0],    # Yellow
            [1, 0, 1],    # Magenta
            [0, 1, 1],    # Cyan
            [1, 0.5, 0],  # Orange
            [0.5, 0, 1],  # Purple
        ]
        
        for i, plane in enumerate(planes):
            # Color the plane point cloud
            plane_cloud = plane['cloud'].__copy__()
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
        
        o3d.visualization.draw_geometries(geometries, window_name=window_name)
    
    @staticmethod
    def visualize_poses(point_clouds: List[o3d.geometry.PointCloud],
                       poses: List[Dict],
                       box_dimensions: np.ndarray,
                       window_name: str = "Estimated Poses"):
        """
        Visualize estimated poses with coordinate frames and bounding boxes.
        
        Args:
            point_clouds: List of segmented box point clouds
            poses: List of estimated pose dictionaries
            box_dimensions: Box dimensions [length, width, height]
            window_name: Window title
        """
        geometries = []
        
        # Add point clouds
        for i, pcd in enumerate(point_clouds):
            pcd_copy = pcd.__copy__()
            # Color each box differently
            color = np.random.rand(3)
            pcd_copy.paint_uniform_color(color)
            geometries.append(pcd_copy)
        
        # Add pose visualizations
        for pose in poses:
            # Add coordinate frame
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=0.1
            )
            frame.transform(pose['transformation'])
            geometries.append(frame)
            
            # Add box mesh
            box = o3d.geometry.TriangleMesh.create_box(
                width=box_dimensions[0],
                height=box_dimensions[1],
                depth=box_dimensions[2]
            )
            box.translate(-box_dimensions / 2)
            box.transform(pose['transformation'])
            box.paint_uniform_color([0.7, 0.7, 0.7])
            box.compute_vertex_normals()
            geometries.append(box)
        
        # Add world coordinate frame
        world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.2, origin=[0, 0, 0]
        )
        geometries.append(world_frame)
        
        o3d.visualization.draw_geometries(geometries, window_name=window_name)
    
    @staticmethod
    def create_comparison_view(images: List[np.ndarray],
                               titles: List[str],
                               save_path: Optional[str] = None) -> np.ndarray:
        """
        Create a side-by-side comparison of multiple images.
        
        Args:
            images: List of images to compare
            titles: List of titles for each image
            save_path: Optional path to save the comparison
            
        Returns:
            Combined comparison image
        """
        n_images = len(images)
        
        # Create figure
        fig, axes = plt.subplots(1, n_images, figsize=(6 * n_images, 6))
        
        if n_images == 1:
            axes = [axes]
        
        for ax, img, title in zip(axes, images, titles):
            if len(img.shape) == 3:
                # BGR to RGB for matplotlib
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                # Grayscale
                ax.imshow(img, cmap='gray')
                ax.set_title(title)
                ax.axis('off')
                continue
            
            ax.imshow(img)
            ax.set_title(title)
            ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
        
        return fig
    
    @staticmethod
    def print_pose_info(pose: Dict, box_id: int = 0):
        """
        Print pose information in a readable format.
        
        Args:
            pose: Pose dictionary
            box_id: ID of the box
        """
        print(f"\n{'='*50}")
        print(f"Box {box_id} Pose Information")
        print(f"{'='*50}")
        print(f"Position (x, y, z): {pose['position']}")
        print(f"Euler Angles (deg): {pose['euler_angles']}")
        print(f"Quaternion (x,y,z,w): {pose['quaternion']}")
        
        if 'dimensions' in pose:
            print(f"Dimensions: {pose['dimensions']}")
        
        if 'fitness' in pose:
            fitness = pose['fitness']
            rmse = pose['inlier_rmse']
            print(f"ICP Fitness: {fitness:.6f} ({fitness*100:.2f}% inliers)")
            print(f"ICP RMSE: {rmse:.8f} meters")
            
            # Quality indicator
            if fitness > 0.8 and rmse < 0.001:
                quality = "Excellent"
            elif fitness > 0.5 and rmse < 0.005:
                quality = "Good"
            elif fitness > 0.3:
                quality = "Fair"
            else:
                quality = "Poor"
            print(f"Alignment Quality: {quality}")
        
        if 'method' in pose:
            print(f"Estimation Method: {pose['method']}")
        
        print(f"{'='*50}\n")
    
    @staticmethod
    def save_results_to_file(poses: List[Dict], filepath: str):
        """
        Save pose estimation results to a text file.
        
        Args:
            poses: List of pose dictionaries
            filepath: Output file path
        """
        with open(filepath, 'w') as f:
            f.write("Box Pose Estimation Results\n")
            f.write("=" * 80 + "\n\n")
            
            for pose in poses:
                box_id = pose.get('id', 0)
                f.write(f"Box ID: {box_id}\n")
                f.write(f"Position (x, y, z): {pose['position']}\n")
                f.write(f"Euler Angles (deg): {pose['euler_angles']}\n")
                f.write(f"Quaternion (x,y,z,w): {pose['quaternion']}\n")
                
                if 'dimensions' in pose:
                    f.write(f"Dimensions: {pose['dimensions']}\n")
                
                if 'fitness' in pose:
                    fitness = pose['fitness']
                    rmse = pose['inlier_rmse']
                    f.write(f"ICP Fitness: {fitness:.6f} ({fitness*100:.2f}% inliers)\n")
                    f.write(f"ICP RMSE: {rmse:.8f} meters\n")
                
                f.write("\nTransformation Matrix:\n")
                f.write(str(pose['transformation']))
                f.write("\n\n" + "-" * 80 + "\n\n")
    
    @staticmethod
    def visualize_depth_image(depth_image: np.ndarray,
                             save_path: Optional[str] = None) -> np.ndarray:
        """
        Visualize depth image with colormap.

        Args:
            depth_image: Depth image array
            save_path: Optional path to save visualization

        Returns:
            Colorized depth image
        """
        # Normalize depth for visualization
        depth_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
        depth_normalized = depth_normalized.astype(np.uint8)

        # Apply colormap
        depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

        if save_path:
            cv2.imwrite(save_path, depth_colored)

        return depth_colored

    @staticmethod
    def project_3d_to_2d(point_3d: np.ndarray, camera_matrix: np.ndarray) -> Tuple[int, int, float]:
        """
        Project a 3D point to 2D image coordinates using camera intrinsics.

        Args:
            point_3d: 3D point [x, y, z]
            camera_matrix: 3x3 camera intrinsic matrix

        Returns:
            Tuple of (u, v, depth) - image coordinates and depth
        """
        # Project 3D point to 2D
        point_2d_homogeneous = camera_matrix @ point_3d

        # Convert from homogeneous coordinates
        depth = point_2d_homogeneous[2]
        if depth > 0:  # Point is in front of camera
            u = int(point_2d_homogeneous[0] / depth)
            v = int(point_2d_homogeneous[1] / depth)
        else:
            u, v = -1, -1  # Behind camera

        return u, v, depth

    @staticmethod
    def draw_pose_axes_2d(image: np.ndarray, pose: Dict, camera_intrinsics: Dict,
                         axis_length: float = 0.15, line_thickness: int = 4) -> np.ndarray:
        """
        Draw 3D pose axes projected onto a 2D image.

        Args:
            image: Input color image
            pose: Pose dictionary with transformation matrix
            camera_intrinsics: Camera intrinsics dictionary with fx, fy, cx, cy
            axis_length: Length of axes in meters (default 0.15m = 15cm)
            line_thickness: Thickness of drawn lines (default 4)

        Returns:
            Image with drawn pose axes
        """
        result = image.copy()
        height, width = image.shape[:2]

        # Build camera matrix
        K = np.array([
            [camera_intrinsics['fx'], 0, camera_intrinsics['cx']],
            [0, camera_intrinsics['fy'], camera_intrinsics['cy']],
            [0, 0, 1]
        ])

        # Get pose transformation
        T = pose['transformation']

        # Define axes in 3D (origin and endpoints)
        origin = T[:3, 3]  # Origin at pose position
        x_axis = origin + T[:3, 0] * axis_length  # Red: X-axis
        y_axis = origin + T[:3, 1] * axis_length  # Green: Y-axis
        z_axis = origin + T[:3, 2] * axis_length  # Blue: Z-axis

        # Project points to 2D
        origin_u, origin_v, origin_depth = Visualizer.project_3d_to_2d(origin, K)
        x_u, x_v, x_depth = Visualizer.project_3d_to_2d(x_axis, K)
        y_u, y_v, y_depth = Visualizer.project_3d_to_2d(y_axis, K)
        z_u, z_v, z_depth = Visualizer.project_3d_to_2d(z_axis, K)

        # Helper function to clip line to image bounds
        def clip_to_bounds(p1, p2):
            """Clip line to image bounds."""
            x1, y1 = p1
            x2, y2 = p2

            # Simple clipping - just check if at least one point is visible
            p1_visible = (0 <= x1 < width) and (0 <= y1 < height)
            p2_visible = (0 <= x2 < width) and (0 <= y2 < height)

            return p1_visible or p2_visible

        # Draw axes only if origin is in front of camera and visible
        if origin_depth > 0:
            origin_2d = (origin_u, origin_v)

            # Check if origin is roughly in bounds (allow some margin)
            if -width < origin_u < width*2 and -height < origin_v < height*2:

                # Draw X-axis (Red) if endpoint is in front of camera
                if x_depth > 0:
                    x_2d = (x_u, x_v)
                    if clip_to_bounds(origin_2d, x_2d):
                        cv2.arrowedLine(result, origin_2d, x_2d, (0, 0, 255), line_thickness, tipLength=0.2)

                # Draw Y-axis (Green) if endpoint is in front of camera
                if y_depth > 0:
                    y_2d = (y_u, y_v)
                    if clip_to_bounds(origin_2d, y_2d):
                        cv2.arrowedLine(result, origin_2d, y_2d, (0, 255, 0), line_thickness, tipLength=0.2)

                # Draw Z-axis (Blue) if endpoint is in front of camera
                if z_depth > 0:
                    z_2d = (z_u, z_v)
                    if clip_to_bounds(origin_2d, z_2d):
                        cv2.arrowedLine(result, origin_2d, z_2d, (255, 0, 0), line_thickness, tipLength=0.2)

                # Draw origin point (larger and more visible)
                if 0 <= origin_u < width and 0 <= origin_v < height:
                    cv2.circle(result, origin_2d, 8, (255, 255, 255), -1)
                    cv2.circle(result, origin_2d, 10, (0, 0, 0), 2)

        return result

    @staticmethod
    def visualize_poses_2d(image: np.ndarray, poses: List[Dict],
                          camera_intrinsics: Dict,
                          save_path: Optional[str] = None,
                          axis_length: float = 0.15,
                          line_thickness: int = 5) -> np.ndarray:
        """
        Visualize multiple 6D poses on a 2D image.

        Args:
            image: Input color image
            poses: List of pose dictionaries
            camera_intrinsics: Camera intrinsics dictionary
            save_path: Optional path to save the visualization
            axis_length: Length of coordinate axes in meters (default 0.15m)
            line_thickness: Thickness of axes lines (default 5)

        Returns:
            Image with visualized poses
        """
        result = image.copy()

        print(f"    Visualizing {len(poses)} poses on 2D image...")

        # Draw each pose
        drawn_count = 0
        for i, pose in enumerate(poses):
            before_hash = hash(result.tobytes())
            result = Visualizer.draw_pose_axes_2d(result, pose, camera_intrinsics,
                                                 axis_length=axis_length,
                                                 line_thickness=line_thickness)
            after_hash = hash(result.tobytes())

            if before_hash != after_hash:
                drawn_count += 1
                print(f"      ✓ Pose {i} axes drawn successfully")
            else:
                print(f"      ⚠ Pose {i} axes not drawn (may be outside view)")

        print(f"    Successfully drew {drawn_count}/{len(poses)} pose visualizations")

        if save_path:
            cv2.imwrite(save_path, result)

        return result
