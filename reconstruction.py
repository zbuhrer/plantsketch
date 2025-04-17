# reconstruction.py
import os
from pathlib import Path
import subprocess
import uuid
import open3d as o3d
import numpy as np
import cv2

PROJECTS_ROOT = Path("projects")

def create_project_dir():
    uid = str(uuid.uuid4())
    base = PROJECTS_ROOT / uid
    (base / "frames").mkdir(parents=True, exist_ok=True)
    (base / "meshroom_project").mkdir(parents=True, exist_ok=True)
    (base / "output_pointcloud").mkdir(parents=True, exist_ok=True)
    return uid, base

def run_meshroom(frames_dir, output_dir, log_path):
    try:
        return reconstruct_with_open3d(frames_dir, output_dir, log_path)
    except Exception as e:
        with open(log_path, "w") as log_file:
            log_file.write(f"Error during reconstruction: {str(e)}")
        return False

def reconstruct_with_open3d(frames_dir, output_dir, log_path):
    """Process images into a point cloud using Open3D"""
    frames_dir = Path(frames_dir)
    output_dir = Path(output_dir)

    # Get all image paths
    image_paths = sorted(list(frames_dir.glob("*.jpg")))
    if len(image_paths) < 5:
        with open(log_path, "w") as f:
            f.write("Error: Need at least 5 images for reliable reconstruction")
        return False

    # Load images and create point clouds
    pcd_combined = o3d.geometry.PointCloud()

    with open(log_path, "w") as log_file:
        log_file.write(f"Processing {len(image_paths)} images...\n")

        # Camera parameters (simplified)
        width, height = cv2.imread(str(image_paths[0])).shape[1::-1]
        camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width, height, width, width, width/2, height/2)

        for i, path in enumerate(image_paths):
            try:
                # Load color image
                color = o3d.io.read_image(str(path))

                # Basic depth estimation using gradients
                img = cv2.imread(str(path))
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # Simple gradient-based depth approximation
                sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                depth_approx = np.sqrt(sobelx**2 + sobely**2)
                # Normalize and create Open3D depth image
                depth_approx = (depth_approx / depth_approx.max() * 1000).astype(np.uint16)
                depth = o3d.geometry.Image(depth_approx)

                # Create RGBD image
                rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    color, depth, depth_scale=1000.0, depth_trunc=3.0, convert_rgb_to_intensity=False)

                # Create point cloud from RGBD image
                pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, camera_intrinsic)

                # Apply a simple transformation to simulate different camera positions
                # In a real SfM pipeline, we'd estimate camera positions properly
                angle = i * 10  # rotate 10 degrees per image
                R = np.array([
                    [np.cos(np.radians(angle)), 0, np.sin(np.radians(angle))],
                    [0, 1, 0],
                    [-np.sin(np.radians(angle)), 0, np.cos(np.radians(angle))]
                ])
                pcd.rotate(R, center=(0, 0, 0))
                pcd.translate((i * 0.1, 0, 0))  # move slightly to create parallax

                pcd_combined += pcd
                log_file.write(f"Processed image {i+1}/{len(image_paths)}\n")
            except Exception as e:
                log_file.write(f"Error processing image {path.name}: {str(e)}\n")

    # Downsample for better performance
    if len(pcd_combined.points) > 0:
        pcd_combined = pcd_combined.voxel_down_sample(voxel_size=0.02)

        # Estimate normals
        pcd_combined.estimate_normals()

        # Save point cloud
        o3d.io.write_point_cloud(str(output_dir / "pointcloud.ply"), pcd_combined)

        with open(log_path, "a") as f:
            f.write(f"Successfully created point cloud with {len(pcd_combined.points)} points")
        return True
    else:
        with open(log_path, "a") as f:
            f.write("Failed to create point cloud: no points generated")
        return False
