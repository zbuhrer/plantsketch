# reconstruction.py
import os
from pathlib import Path
import subprocess
import uuid
import open3d as o3d
import numpy as np
import cv2
import traceback
from datetime import datetime

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
            log_file.write(f"Error during reconstruction: {str(e)}\n")
            log_file.write(traceback.format_exc())
        return False

def reconstruct_with_open3d(frames_dir, output_dir, log_path):
    """Process images into a point cloud using Open3D"""
    frames_dir = Path(frames_dir)
    output_dir = Path(output_dir)

    with open(log_path, "w") as log_file:
        log_file.write(f"[{datetime.now()}] Starting reconstruction from {frames_dir}\n")

        # Get all image paths
        image_paths = sorted(list(frames_dir.glob("*.jpg")) + list(frames_dir.glob("*.jpeg")) +
                            list(frames_dir.glob("*.png")))
        log_file.write(f"[{datetime.now()}] Found {len(image_paths)} images: {[p.name for p in image_paths]}\n")

        if len(image_paths) < 5:
            log_file.write(f"[{datetime.now()}] Error: Need at least 5 images for reliable reconstruction\n")
            return False

        # Load images and create point clouds
        pcd_combined = o3d.geometry.PointCloud()

        log_file.write(f"[{datetime.now()}] Processing {len(image_paths)} images...\n")

        # Camera parameters (simplified)
        try:
            first_img = cv2.imread(str(image_paths[0]))
            if first_img is None:
                log_file.write(f"[{datetime.now()}] Error: Could not read image {image_paths[0]}\n")
                return False

            width, height = first_img.shape[1::-1]
            log_file.write(f"[{datetime.now()}] Image dimensions: {width}x{height}\n")

            camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
                width, height, width, width, width/2, height/2)
            log_file.write(f"[{datetime.now()}] Camera intrinsic matrix set up successfully\n")
        except Exception as e:
            log_file.write(f"[{datetime.now()}] Error setting up camera parameters: {str(e)}\n")
            log_file.write(traceback.format_exc())
            return False

        for i, path in enumerate(image_paths):
            try:
                log_file.write(f"[{datetime.now()}] Processing image {i+1}/{len(image_paths)}: {path.name}\n")

                # Load color image
                color = o3d.io.read_image(str(path))
                log_file.write(f"[{datetime.now()}]   Color image loaded successfully\n")

                # Basic depth estimation using gradients
                img = cv2.imread(str(path))
                if img is None:
                    log_file.write(f"[{datetime.now()}]   Error: Could not read image with OpenCV\n")
                    continue

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # Simple gradient-based depth approximation
                sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                depth_approx = np.sqrt(sobelx**2 + sobely**2)

                # Check if depth_approx has valid values
                if depth_approx.max() == 0:
                    log_file.write(f"[{datetime.now()}]   Error: Depth approximation failed (all zeros)\n")
                    continue

                # Normalize and create Open3D depth image
                depth_approx = (depth_approx / depth_approx.max() * 1000).astype(np.uint16)
                depth = o3d.geometry.Image(depth_approx)
                log_file.write(f"[{datetime.now()}]   Depth image created successfully\n")

                # Create RGBD image
                rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    color, depth, depth_scale=1000.0, depth_trunc=3.0, convert_rgb_to_intensity=False)
                log_file.write(f"[{datetime.now()}]   RGBD image created successfully\n")

                # Create point cloud from RGBD image
                pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, camera_intrinsic)
                log_file.write(f"[{datetime.now()}]   Point cloud created with {len(pcd.points)} points\n")

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
                log_file.write(f"[{datetime.now()}]   Added to combined point cloud successfully\n")
            except Exception as e:
                log_file.write(f"[{datetime.now()}] Error processing image {path.name}: {str(e)}\n")
                log_file.write(traceback.format_exc())

        # Report combined point cloud stats
        log_file.write(f"[{datetime.now()}] Combined point cloud has {len(pcd_combined.points)} points\n")

        # Downsample for better performance
        if len(pcd_combined.points) > 0:
            try:
                log_file.write(f"[{datetime.now()}] Starting downsampling process\n")
                pcd_combined = pcd_combined.voxel_down_sample(voxel_size=0.02)
                log_file.write(f"[{datetime.now()}] Downsampled to {len(pcd_combined.points)} points\n")

                # Estimate normals
                log_file.write(f"[{datetime.now()}] Estimating normals\n")
                pcd_combined.estimate_normals()
                log_file.write(f"[{datetime.now()}] Normals estimated successfully\n")

                # Save point cloud
                output_path = str(output_dir / "pointcloud.ply")
                log_file.write(f"[{datetime.now()}] Saving point cloud to {output_path}\n")
                o3d.io.write_point_cloud(output_path, pcd_combined)
                log_file.write(f"[{datetime.now()}] Point cloud saved successfully\n")

                log_file.write(f"[{datetime.now()}] Successfully created point cloud with {len(pcd_combined.points)} points")
                return True
            except Exception as e:
                log_file.write(f"[{datetime.now()}] Error in final processing: {str(e)}\n")
                log_file.write(traceback.format_exc())
                return False
        else:
            log_file.write(f"[{datetime.now()}] Failed to create point cloud: no points generated\n")
            return False
