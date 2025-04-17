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

def estimate_depth(img, log_file):
    """Improved depth estimation with better filtering"""
    log_file.write(f"[{datetime.now()}] Estimating depth with improved method\n")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Compute gradients with Sobel (larger kernel for better results)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)

    # Combine gradients
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)

    # Apply additional filtering for smoother depth
    gradient_magnitude = cv2.bilateralFilter(
        gradient_magnitude.astype(np.float32), 9, 75, 75)

    # Check if depth estimation succeeded
    if gradient_magnitude.max() == 0:
        log_file.write(f"[{datetime.now()}] Error: Depth approximation failed (all zeros)\n")
        return None

    # Normalize to proper depth range
    depth = (gradient_magnitude / np.max(gradient_magnitude) * 1000).astype(np.uint16)
    return depth

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
        pcds = []  # Store individual point clouds for registration
        pcd_combined = o3d.geometry.PointCloud()

        log_file.write(f"[{datetime.now()}] Processing {len(image_paths)} images...\n")

        # Camera parameters with better estimates for smartphone cameras
        try:
            first_img = cv2.imread(str(image_paths[0]))
            if first_img is None:
                log_file.write(f"[{datetime.now()}] Error: Could not read image {image_paths[0]}\n")
                return False

            width, height = first_img.shape[1::-1]
            log_file.write(f"[{datetime.now()}] Image dimensions: {width}x{height}\n")

            # Better camera intrinsic parameters for smartphone camera
            fx = fy = width * 1.2  # Better focal length estimate for typical smartphone
            cx, cy = width/2, height/2  # Principal point
            camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
            log_file.write(f"[{datetime.now()}] Camera intrinsic matrix set up with smartphone estimates\n")
            log_file.write(f"[{datetime.now()}] Focal length: fx={fx}, fy={fy}, principal point: ({cx}, {cy})\n")
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

                # Use improved depth estimation
                img = cv2.imread(str(path))
                if img is None:
                    log_file.write(f"[{datetime.now()}]   Error: Could not read image with OpenCV\n")
                    continue

                depth_approx = estimate_depth(img, log_file)
                if depth_approx is None:
                    continue

                depth = o3d.geometry.Image(depth_approx)
                log_file.write(f"[{datetime.now()}]   Depth image created successfully\n")

                # Create RGBD image
                rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    color, depth, depth_scale=1000.0, depth_trunc=3.0, convert_rgb_to_intensity=False)
                log_file.write(f"[{datetime.now()}]   RGBD image created successfully\n")

                # Create point cloud from RGBD image
                pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, camera_intrinsic)
                log_file.write(f"[{datetime.now()}]   Point cloud created with {len(pcd.points)} points\n")

                # Store individual point cloud for registration
                pcds.append(pcd)

            except Exception as e:
                log_file.write(f"[{datetime.now()}] Error processing image {path.name}: {str(e)}\n")
                log_file.write(traceback.format_exc())

        # Perform point cloud registration when we have enough point clouds
        if len(pcds) >= 2:
            log_file.write(f"[{datetime.now()}] Performing point cloud registration\n")
            try:
                # First point cloud is our reference
                pcd_combined = pcds[0]

                # Register each subsequent point cloud to the combined one
                for i in range(1, len(pcds)):
                    if len(pcds[i].points) < 10:
                        log_file.write(f"[{datetime.now()}]   Skipping point cloud {i} - too few points\n")
                        continue

                    log_file.write(f"[{datetime.now()}]   Registering point cloud {i} to combined cloud\n")

                    # Both point clouds need normals for point-to-plane ICP
                    pcd_combined.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
                    pcds[i].estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

                    # Point-to-point ICP registration (more robust than point-to-plane for our case)
                    result = o3d.pipelines.registration.registration_icp(
                        pcds[i], pcd_combined, 0.05, np.identity(4),
                        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100))

                    log_file.write(f"[{datetime.now()}]   Registration fitness: {result.fitness}\n")

                    # Transform the point cloud based on registration result
                    pcds[i].transform(result.transformation)

                    # Add to combined point cloud
                    pcd_combined += pcds[i]
            except Exception as e:
                log_file.write(f"[{datetime.now()}] Error during point cloud registration: {str(e)}\n")
                log_file.write(traceback.format_exc())
                # Continue with available point clouds even if registration fails
        else:
            log_file.write(f"[{datetime.now()}] Not enough valid point clouds for registration\n")
            if len(pcds) == 1:
                pcd_combined = pcds[0]

        # Report combined point cloud stats
        if pcd_combined and len(pcd_combined.points) > 0:
            log_file.write(f"[{datetime.now()}] Combined point cloud has {len(pcd_combined.points)} points\n")
        else:
            log_file.write(f"[{datetime.now()}] Combined point cloud is empty\n")
            return False

        # Downsample for better performance
        if len(pcd_combined.points) > 0:
            try:
                log_file.write(f"[{datetime.now()}] Starting downsampling process\n")
                pcd_combined = pcd_combined.voxel_down_sample(voxel_size=0.02)
                log_file.write(f"[{datetime.now()}] Downsampled to {len(pcd_combined.points)} points\n")

                # Estimate normals with increased search radius for better results
                log_file.write(f"[{datetime.now()}] Estimating normals\n")
                pcd_combined.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
                pcd_combined.orient_normals_consistent_tangent_plane(20)  # Improved normal orientation
                log_file.write(f"[{datetime.now()}] Normals estimated successfully\n")

                # Statistical outlier removal for cleaner point cloud
                log_file.write(f"[{datetime.now()}] Removing outliers\n")
                pcd_combined, _ = pcd_combined.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
                log_file.write(f"[{datetime.now()}] After outlier removal: {len(pcd_combined.points)} points\n")

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
