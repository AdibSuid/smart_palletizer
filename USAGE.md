# Usage Guide

This guide provides detailed instructions on how to use the Smart Palletizer package.

## Installation

### Method 1: Using the install script (Recommended)

```bash
cd smart_palletizer
./install.sh
```

### Method 2: Manual installation

```bash
# Create virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate  # On Linux/Mac
# or
venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Verify Installation

```bash
python test_installation.py
```

## Running the Pipeline

### Run All Tasks

Process all four tasks on small boxes:

```bash
python -m smart_palletizier.pipeline --data-folder data/small_box --box-type small_box
```

Process medium boxes:

```bash
python -m smart_palletizier.pipeline --data-folder data/medium_box --box-type medium_box
```

### Run Specific Tasks

Run only Task 1 (2D detection) and Task 4 (pose estimation):

```bash
python -m smart_palletizier.pipeline --data-folder data/small_box --box-type small_box --tasks 1,4
```

Run only Task 2 (planar patches):

```bash
python -m smart_palletizier.pipeline --tasks 2
```

## Using Individual Modules

### Task 1: 2D Box Detection

```python
import cv2
from smart_palletizier import BoxDetector

# Initialize detector
detector = BoxDetector(min_area=1000, max_area=500000)

# Load image
image = cv2.imread("data/small_box/color_image.png")

# Method 1: Detect from image
boxes = detector.detect(image)

# Method 2: Detect from mask (more accurate)
mask = cv2.imread("data/small_box/small_box_mask_0.png", cv2.IMREAD_GRAYSCALE)
boxes = detector.detect_boxes_from_mask(mask)

# Visualize results
result = detector.draw_detections(image, boxes, "small_box")
cv2.imwrite("detection_result.png", result)

# Access box information
for box in boxes:
    print(f"Box {box['id']}:")
    print(f"  Center: {box['center']}")
    print(f"  Bounding box: {box['bbox']}")  # (x, y, width, height)
    print(f"  Area: {box['area']}")
```

### Task 2: Planar Patches Detection

```python
from smart_palletizier import PointCloudProcessor, PlanarPatchDetector

# Initialize
pcd_processor = PointCloudProcessor()
planar_detector = PlanarPatchDetector(distance_threshold=0.005, min_points=50)

# Load and clean point cloud
pcd = pcd_processor.load_pointcloud("data/small_box/small_box_0_raw.ply")
cleaned = pcd_processor.clean_pointcloud(pcd)

# Detect planes
planes = planar_detector.detect_planes(cleaned, max_planes=10)

print(f"Detected {len(planes)} planes")

# Group planes by orientation (X, Y, Z axes)
oriented_groups = planar_detector.group_planes_by_orientation(planes)
print(f"X-aligned: {len(oriented_groups['X'])}")
print(f"Y-aligned: {len(oriented_groups['Y'])}")
print(f"Z-aligned: {len(oriented_groups['Z'])}")

# Classify box faces
faces = planar_detector.classify_box_faces(planes)
print(f"Top face: {faces['top'] is not None}")
print(f"Bottom face: {faces['bottom'] is not None}")

# Analyze complete box structure
analysis = planar_detector.analyze_box_from_planes(planes)
print(f"Box center: {analysis['center']}")
print(f"Visible faces: {analysis['num_visible_faces']}")
print(f"Dimensions: {analysis['dimensions']}")

# Visualize
from smart_palletizier import Visualizer
Visualizer.visualize_planar_patches(planes, cleaned)
```

### Task 3: Point Cloud Post-Processing

```python
from smart_palletizier import PointCloudProcessor

# Initialize with custom parameters
processor = PointCloudProcessor(
    voxel_size=0.002,      # Downsample voxel size (meters)
    nb_neighbors=20,        # Neighbors for statistical filtering
    std_ratio=2.0           # Standard deviation multiplier
)

# Load point cloud
pcd = processor.load_pointcloud("data/small_box/small_box_0_raw.ply")
print(f"Original: {len(pcd.points)} points")

# Get original dimensions
dims_before = processor.get_dimensions(pcd)
print(f"Dimensions before: {dims_before}")

# Clean point cloud (full pipeline)
cleaned = processor.clean_pointcloud(
    pcd,
    downsample=True,
    remove_outliers=True
)
print(f"Cleaned: {len(cleaned.points)} points")

# Get cleaned dimensions
dims_after = processor.get_dimensions(cleaned)
print(f"Dimensions after: {dims_after}")

# Individual processing steps
cleaned = pcd
cleaned = processor.remove_statistical_outliers(cleaned)
cleaned = processor.remove_radius_outliers(cleaned, radius=0.01, min_points=5)
cleaned = processor.downsample(cleaned, voxel_size=0.002)
processor.estimate_normals(cleaned)

# Save processed cloud
processor.save_pointcloud(cleaned, "cleaned_output.ply")

# Visualize
from smart_palletizier import Visualizer
Visualizer.visualize_point_cloud(cleaned, "Cleaned Point Cloud")
```

### Task 4: 6D Pose Estimation

```python
import numpy as np
from smart_palletizier import (
    PointCloudProcessor,
    PoseEstimator,
    Visualizer
)

# Initialize
pcd_processor = PointCloudProcessor()
pose_estimator = PoseEstimator(
    icp_threshold=0.002,
    icp_max_iterations=2000
)

# Box dimensions (meters)
box_dimensions = np.array([0.340, 0.250, 0.095])  # small box

# Load and clean point cloud
pcd = pcd_processor.load_pointcloud("data/small_box/small_box_0_raw.ply")
cleaned = pcd_processor.clean_pointcloud(pcd)

# Load template mesh (optional, improves accuracy)
template = pose_estimator.load_box_mesh("data/small_box/small_box_mesh.ply")

# Estimate pose for single box
poses = pose_estimator.estimate_multiple_boxes(
    [cleaned],
    template,
    box_dimensions
)

# Access pose information
pose = poses[0]
print(f"Position (x,y,z): {pose['position']}")
print(f"Euler angles (deg): {pose['euler_angles']}")
print(f"Quaternion: {pose['quaternion']}")
print(f"ICP Fitness: {pose.get('fitness', 'N/A')}")
print(f"ICP RMSE: {pose.get('inlier_rmse', 'N/A')}")

# Get transformation matrix
transform = pose['transformation']
print(f"4x4 Transform:\n{transform}")

# Create visualization meshes
box_mesh = pose_estimator.create_box_mesh(box_dimensions, pose)
coord_frame = pose_estimator.create_coordinate_frame(pose, size=0.1)

# Visualize
Visualizer.visualize_poses([cleaned], poses, box_dimensions)

# Save results
Visualizer.save_results_to_file(poses, "pose_results.txt")
```

## Advanced Usage

### Batch Processing Multiple Boxes

```python
from pathlib import Path
from smart_palletizier import PointCloudProcessor, PoseEstimator
import numpy as np

processor = PointCloudProcessor()
pose_estimator = PoseEstimator()

# Load all box point clouds
data_folder = Path("data/small_box")
pcd_files = list(data_folder.glob("small_box_*_raw.ply"))

# Clean all clouds
cleaned_clouds = []
for pcd_file in pcd_files:
    pcd = processor.load_pointcloud(str(pcd_file))
    cleaned = processor.clean_pointcloud(pcd)
    cleaned_clouds.append(cleaned)

# Estimate all poses
box_dimensions = np.array([0.340, 0.250, 0.095])
template = pose_estimator.load_box_mesh(str(data_folder / "small_box_mesh.ply"))
poses = pose_estimator.estimate_multiple_boxes(
    cleaned_clouds,
    template,
    box_dimensions
)

# Process results
for pose in poses:
    print(f"Box {pose['id']}: position={pose['position']}")
```

### Creating Custom Visualizations

```python
import open3d as o3d
from smart_palletizier import Visualizer

# Create comparison view
import cv2
image1 = cv2.imread("image1.png")
image2 = cv2.imread("image2.png")

Visualizer.create_comparison_view(
    [image1, image2],
    ["Before", "After"],
    save_path="comparison.png"
)

# Visualize depth image
depth = cv2.imread("depth.png", cv2.IMREAD_UNCHANGED)
depth_colored = Visualizer.visualize_depth_image(depth, "depth_viz.png")

# Custom 3D visualization
geometries = [point_cloud, mesh, coordinate_frame]
Visualizer.visualize_multiple_geometries(
    geometries,
    window_name="Custom View"
)
```

### Parameter Tuning

#### Box Detector Parameters

```python
detector = BoxDetector(
    min_area=1000,           # Minimum contour area (pixels)
    max_area=500000,         # Maximum contour area (pixels)
    aspect_ratio_range=(0.3, 3.0)  # Valid aspect ratios
)
```

#### Point Cloud Processor Parameters

```python
processor = PointCloudProcessor(
    voxel_size=0.002,       # Smaller = more detail, slower
    nb_neighbors=20,         # More = smoother, may lose detail
    std_ratio=2.0           # Higher = keep more points
)
```

#### Planar Detector Parameters

```python
planar_detector = PlanarPatchDetector(
    distance_threshold=0.005,  # Max distance to plane (meters)
    ransac_n=3,                # Points per RANSAC iteration
    num_iterations=1000,       # RANSAC iterations
    min_points=50              # Min points for valid plane
)
```

#### Pose Estimator Parameters

```python
pose_estimator = PoseEstimator(
    icp_threshold=0.002,        # ICP distance threshold
    icp_max_iterations=2000     # ICP max iterations
)
```

## Troubleshooting

### Import Errors

If you get import errors, ensure all dependencies are installed:

```bash
pip install -r requirements.txt
```

### Open3D Visualization Issues

If visualization windows don't appear:

```python
import open3d as o3d
# Check if Open3D can create a window
print(o3d.__version__)
```

### Point Cloud Empty After Cleaning

Try less aggressive filtering:

```python
processor = PointCloudProcessor(
    voxel_size=0.005,  # Larger voxel
    std_ratio=3.0      # Keep more points
)
```

### Poor ICP Results

- Ensure point clouds are cleaned properly
- Check that template mesh matches box type
- Verify box dimensions are correct
- Try adjusting ICP threshold

## Output Files

The pipeline creates the following outputs in the data folder:

- `*_detection_result.png` - 2D detection visualization
- `*_cleaned.ply` - Cleaned point clouds
- `*_pose_results.txt` - Pose estimation results

## Next Steps

- Read [METHODOLOGY.md](METHODOLOGY.md) for detailed explanations
- Run `python examples/demo.py` for more examples
- Check the API documentation in source files (docstrings)
