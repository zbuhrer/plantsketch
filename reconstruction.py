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

def estimate_depth(img, log_func):
    """Improved depth estimation with better filtering"""
    log_func("Estimating depth with improved method")
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
        log_func("Error: Depth approximation failed (all zeros)")
        return None

    # Normalize to proper depth range
    depth = (gradient_magnitude / np.max(gradient_magnitude) * 1000).astype(np.uint16)
    return depth

def reconstruct_with_open3d(frames_dir, output_dir, log_path, log_display=None, log_content=""):
    """Process images into a point cloud using Open3D with real-time logging"""
    frames_dir = Path(frames_dir)
    output_dir = Path(output_dir)

    # Initialize tracking for registration progress
    registration_data = []

    def log(message):
        """Helper function to log messages to both file and UI"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"

        # Write to log file
        with open(log_path, "a") as log_file:
            log_file.write(log_entry)

        # Update the UI if log_display is provided
        nonlocal log_content
        if log_display is not None:
            log_content += log_entry
            log_display.code(log_content, language="text")

    # Initialize log file
    with open(log_path, "w") as _:
        pass  # Create empty file

    log(f"Starting reconstruction from {frames_dir}")

    # Get all image paths
    image_paths = sorted(list(frames_dir.glob("*.jpg")) + list(frames_dir.glob("*.jpeg")) +
                        list(frames_dir.glob("*.png")))
    log(f"Found {len(image_paths)} images: {[p.name for p in image_paths]}")

    if len(image_paths) < 5:
        log("Error: Need at least 5 images for reliable reconstruction")
        return (False, log_content) if log_display else False

    # Load images and create point clouds
    pcds = []  # Store individual point clouds for registration
    pcd_combined = o3d.geometry.PointCloud()

    log(f"Processing {len(image_paths)} images...")

    # Camera parameters with better estimates for smartphone cameras
    try:
        first_img = cv2.imread(str(image_paths[0]))
        if first_img is None:
            log(f"Error: Could not read image {image_paths[0]}")
            return (False, log_content) if log_display else False

        width, height = first_img.shape[1::-1]
        log(f"Image dimensions: {width}x{height}")

        # Better camera intrinsic parameters for smartphone camera
        fx = fy = width * 1.2  # Better focal length estimate for typical smartphone
        cx, cy = width/2, height/2  # Principal point
        camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
        log("Camera intrinsic matrix set up with smartphone estimates")
        log(f"Focal length: fx={fx}, fy={fy}, principal point: ({cx}, {cy})")
    except Exception as e:
        log(f"Error setting up camera parameters: {str(e)}")
        log(traceback.format_exc())
        return (False, log_content) if log_display else False

    for i, path in enumerate(image_paths):
        try:
            log(f"Processing image {i+1}/{len(image_paths)}: {path.name}")

            # Load color image
            color = o3d.io.read_image(str(path))
            log("  Color image loaded successfully")

            # Use improved depth estimation
            img = cv2.imread(str(path))
            if img is None:
                log("  Error: Could not read image with OpenCV")
                continue

            depth_approx = estimate_depth(img, log)
            if depth_approx is None:
                continue

            depth = o3d.geometry.Image(depth_approx)
            log("  Depth image created successfully")

            # Create RGBD image
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color, depth, depth_scale=1000.0, depth_trunc=3.0, convert_rgb_to_intensity=False)
            log("  RGBD image created successfully")

            # Create point cloud from RGBD image
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, camera_intrinsic)
            log(f"  Point cloud created with {len(pcd.points)} points")

            # Store individual point cloud for registration
            pcds.append(pcd)

        except Exception as e:
            log(f"Error processing image {path.name}: {str(e)}")
            log(traceback.format_exc())

    # Perform point cloud registration when we have enough point clouds
    if len(pcds) >= 2:
        log("Performing point cloud registration")
        try:
            # First point cloud is our reference
            pcd_combined = pcds[0]
            total_to_register = len(pcds) - 1
            registered_count = 0

            # Register each subsequent point cloud to the combined one
            for i in range(1, len(pcds)):
                start_time = datetime.now()

                if len(pcds[i].points) < 10:
                    log(f"  Skipping point cloud {i} - too few points")
                    continue

                log(f"  Registering point cloud {i} to combined cloud")

                # Both point clouds need normals for point-to-plane ICP
                pcd_combined.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
                pcds[i].estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

                # Point-to-point ICP registration (more robust than point-to-plane for our case)
                result = o3d.pipelines.registration.registration_icp(
                    pcds[i], pcd_combined, 0.05, np.identity(4),
                    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100))

                log(f"  Registration fitness: {result.fitness}")

                # Calculate processing time
                processing_time = (datetime.now() - start_time).total_seconds()

                # Transform the point cloud based on registration result
                pcds[i].transform(result.transformation)

                # Add to combined point cloud
                pcd_combined += pcds[i]

                # Update registration tracking
                registered_count += 1
                progress_percentage = (registered_count / total_to_register) * 100

                # Store metrics for this registration
                registration_data.append({
                    'index': i,
                    'timestamp': datetime.now().strftime("%H:%M:%S"),
                    'points': len(pcds[i].points),
                    'fitness': float(result.fitness),
                    'processing_time': processing_time,
                    'percent_complete': progress_percentage
                })

                log(f"  Progress: {progress_percentage:.1f}% ({registered_count}/{total_to_register})")
        except Exception as e:
            log(f"Error during point cloud registration: {str(e)}")
            log(traceback.format_exc())
            # Continue with available point clouds even if registration fails
    else:
        log("Not enough valid point clouds for registration")
        if len(pcds) == 1:
            pcd_combined = pcds[0]

    # Report combined point cloud stats
    if pcd_combined and len(pcd_combined.points) > 0:
        log(f"Combined point cloud has {len(pcd_combined.points)} points")
    else:
        log("Combined point cloud is empty")
        return (False, log_content, registration_data) if log_display else (False, registration_data)

    # Downsample for better performance
    if len(pcd_combined.points) > 0:
        try:
            log("Starting downsampling process")
            pcd_combined = pcd_combined.voxel_down_sample(voxel_size=0.02)
            log(f"Downsampled to {len(pcd_combined.points)} points")

            # Estimate normals with increased search radius for better results
            log("Estimating normals")
            pcd_combined.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            pcd_combined.orient_normals_consistent_tangent_plane(20)  # Improved normal orientation
            log("Normals estimated successfully")

            # Statistical outlier removal for cleaner point cloud
            log("Removing outliers")
            pcd_combined, _ = pcd_combined.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            log(f"After outlier removal: {len(pcd_combined.points)} points")

            # Save point cloud
            output_path = str(output_dir / "pointcloud.ply")
            log(f"Saving point cloud to {output_path}")
            o3d.io.write_point_cloud(output_path, pcd_combined)
            log("Point cloud saved successfully")

            log(f"Successfully created point cloud with {len(pcd_combined.points)} points")
            return (True, log_content, registration_data) if log_display else (True, registration_data)
        except Exception as e:
            log(f"Error in final processing: {str(e)}")
            log(traceback.format_exc())
            return (False, log_content, registration_data) if log_display else (False, registration_data)
    else:
        log("Failed to create point cloud: no points generated")
        return (False, log_content, registration_data) if log_display else (False, registration_data)
