# app.py
import streamlit as st
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

from db import init_db, insert_scan, get_all_scans
from extract_frames import extract_frames
from reconstruction import create_project_dir, reconstruct_with_open3d

init_db()
st.title("Plantsketch")

input_method = st.radio(
    "Choose input method",
    ["Video", "Images"]
)

if input_method == "Video":
    video_file = st.file_uploader("Upload a garden video", type=["mp4", "mov", "avi"])
    fps = st.slider("FPS to extract", 1, 10, 2)

    if video_file:
        st.success("Video received.")
        if st.button("Start Scan with Video"):
            uuid, base = create_project_dir()

            # Preserve original video extension
            extension = video_file.name.split('.')[-1].lower()
            video_path = base / f"input_video.{extension}"
            with open(video_path, "wb") as f:
                f.write(video_file.read())

            st.write("Extracting frames...")
            frame_count = extract_frames(video_path, base / "frames", fps=fps)

            # Create placeholder for real-time logging
            st.write("## Processing Status")
            col1, col2 = st.columns(2)

            with col1:
                log_display = st.empty()
                log_content = "Starting reconstruction process...\n"
                log_display.code(log_content, language="text")

            with col2:
                st.write("### Registration Progress")
                progress_placeholder = st.empty()
                chart_placeholder = st.empty()

            st.write("Running 3D reconstruction...")
            log_path = base / "reconstruction_log.txt"

            # Run reconstruction
            success, log_content, registration_data = reconstruct_with_open3d(
                base / "frames",
                base / "output_pointcloud",
                log_path,
                log_display=log_display,
                log_content=log_content
            )

            insert_scan(uuid, video_file.name, frame_count, success)

            if success:
                st.success(f"Scan complete! UUID: {uuid}")

                # Show the log for verification
                with open(log_path, "r") as f:
                    st.expander("Complete Reconstruction Log").code(f.read(), language="text")

                # After completion, show registration statistics
                if registration_data:
                    df = pd.DataFrame(registration_data)

                    # Update final progress
                    if 'percent_complete' in df and len(df) > 0:
                        progress_placeholder.progress(min(float(df['percent_complete'].max()) / 100.0, 1.0))
                    else:
                        progress_placeholder.progress(1.0)

                    st.write("## Registration Summary")
                    summary_col1, summary_col2 = st.columns(2)

                    with summary_col1:
                        st.metric("Total Point Clouds Registered",
                                f"{len(registration_data)} of {len(registration_data) + 1}",
                                f"{len(registration_data)/(len(registration_data) + 1)*100:.1f}%")

                        if 'processing_time' in df:
                            avg_time = df['processing_time'].mean()
                            st.metric("Avg. Registration Time", f"{avg_time:.2f} sec")

                    with summary_col2:
                        if 'fitness' in df:
                            avg_fitness = df['fitness'].mean()
                            st.metric("Avg. Registration Quality", f"{avg_fitness:.4f}")

                        if 'points' in df:
                            total_points = df['points'].sum()
                            st.metric("Total Points Added", f"{total_points:,}")

                    # Create a more detailed final histogram
                    if 'fitness' in df and 'points' in df:
                        fig, axs = plt.subplots(2, 1, figsize=(10, 10))

                        axs[0].plot(df['index'], df['fitness'], 'b-o')
                        axs[0].set_title('Registration Quality by Point Cloud')
                        axs[0].set_ylabel('ICP Fitness Score')
                        axs[0].grid(True)

                        axs[1].bar(df['index'], df['points'])
                        axs[1].set_title('Point Count by Point Cloud')
                        axs[1].set_xlabel('Point Cloud Index')
                        axs[1].set_ylabel('Number of Points')
                        axs[1].grid(True)

                        plt.tight_layout()
                        chart_placeholder.pyplot(fig)

                # Create interactive 3D visualization with Plotly
                try:
                    # Load the point cloud for visualization
                    pcd_path = base / "output_pointcloud" / "pointcloud.ply"
                    if pcd_path.exists():
                        import open3d as o3d
                        pcd = o3d.io.read_point_cloud(str(pcd_path))

                        # Subsample points for visualization (for performance)
                        if len(pcd.points) > 0:
                            st.write("## 3D Point Cloud Visualization")
                            # Downsample more aggressively for larger point clouds
                            sample_rate = max(1, len(pcd.points) // 10000)  # Target ~10k points for viz
                            vis_points = np.asarray(pcd.points)[::sample_rate]

                            # Handle colors if available
                            if pcd.has_colors():
                                vis_colors = np.asarray(pcd.colors)[::sample_rate]
                                # Convert RGB to hex colors for plotly
                                colors = ['rgb({},{},{})'.format(int(r*255), int(g*255), int(b*255))
                                        for r, g, b in vis_colors]
                            else:
                                # Default color if no colors available
                                colors = 'rgb(100,100,100)'

                            # Create plotly figure
                            fig = go.Figure(data=[go.Scatter3d(
                                x=vis_points[:, 0],
                                y=vis_points[:, 1],
                                z=vis_points[:, 2],
                                mode='markers',
                                marker=dict(
                                    size=2,
                                    color=colors,
                                    opacity=0.8
                                )
                            )])

                            fig.update_layout(
                                title=f"Point Cloud Preview ({len(vis_points):,} of {len(pcd.points):,} points)",
                                margin=dict(l=0, r=0, b=0, t=40),
                                scene=dict(
                                    aspectmode='data'
                                )
                            )

                            st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not create 3D visualization: {str(e)}")
            else:
                st.error("Reconstruction failed.")
                # Show error details
                with open(log_path, "r") as f:
                    st.expander("Complete Error Details").code(f.read(), language="text")
else:
    image_files = st.file_uploader("Upload garden images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if image_files:
        st.success(f"{len(image_files)} images received.")
        if st.button("Start Scan with Images"):
            uuid, base = create_project_dir()

            # Save uploaded images to the frames directory
            frames_dir = base / "frames"
            frame_count = 0

            for img in image_files:
                # Extract the file extension to preserve original format
                original_name = img.name
                extension = original_name.split('.')[-1].lower()
                img_path = frames_dir / f"image_{frame_count:04d}.{extension}"
                with open(img_path, "wb") as f:
                    f.write(img.read())
                frame_count += 1

            # Create placeholder for real-time logging
            st.write("## Processing Status")
            col1, col2 = st.columns(2)

            with col1:
                log_display = st.empty()
                log_content = "Starting reconstruction process...\n"
                log_display.code(log_content, language="text")

            with col2:
                st.write("### Registration Progress")
                progress_placeholder = st.empty()
                chart_placeholder = st.empty()

            st.write("Running 3D reconstruction...")
            log_path = base / "reconstruction_log.txt"

            # Run reconstruction
            success, log_content, registration_data = reconstruct_with_open3d(
                frames_dir,
                base / "output_pointcloud",
                log_path,
                log_display=log_display,
                log_content=log_content
            )

            # Use a batch name as reference
            reference_name = f"batch_upload_{len(image_files)}_images"

            insert_scan(uuid, reference_name, frame_count, success)

            if success:
                st.success(f"Scan complete! UUID: {uuid}")

                # Show the log for verification
                with open(log_path, "r") as f:
                    st.expander("Complete Reconstruction Log").code(f.read(), language="text")

                # After completion, show registration statistics
                if registration_data:
                    df = pd.DataFrame(registration_data)

                    # Update final progress
                    if 'percent_complete' in df and len(df) > 0:
                        progress_placeholder.progress(min(float(df['percent_complete'].max()) / 100.0, 1.0))
                    else:
                        progress_placeholder.progress(1.0)

                    st.write("## Registration Summary")
                    summary_col1, summary_col2 = st.columns(2)

                    with summary_col1:
                        st.metric("Total Point Clouds Registered",
                                f"{len(registration_data)} of {len(registration_data) + 1}",
                                f"{len(registration_data)/(len(registration_data) + 1)*100:.1f}%")

                        if 'processing_time' in df:
                            avg_time = df['processing_time'].mean()
                            st.metric("Avg. Registration Time", f"{avg_time:.2f} sec")

                    with summary_col2:
                        if 'fitness' in df:
                            avg_fitness = df['fitness'].mean()
                            st.metric("Avg. Registration Quality", f"{avg_fitness:.4f}")

                        if 'points' in df:
                            total_points = df['points'].sum()
                            st.metric("Total Points Added", f"{total_points:,}")

                    # Create a more detailed final histogram
                    if 'fitness' in df and 'points' in df:
                        fig, axs = plt.subplots(2, 1, figsize=(10, 10))

                        axs[0].plot(df['index'], df['fitness'], 'b-o')
                        axs[0].set_title('Registration Quality by Point Cloud')
                        axs[0].set_ylabel('ICP Fitness Score')
                        axs[0].grid(True)

                        axs[1].bar(df['index'], df['points'])
                        axs[1].set_title('Point Count by Point Cloud')
                        axs[1].set_xlabel('Point Cloud Index')
                        axs[1].set_ylabel('Number of Points')
                        axs[1].grid(True)

                        plt.tight_layout()
                        chart_placeholder.pyplot(fig)

                # Create interactive 3D visualization with Plotly
                try:
                    # Load the point cloud for visualization
                    pcd_path = base / "output_pointcloud" / "pointcloud.ply"
                    if pcd_path.exists():
                        import open3d as o3d
                        pcd = o3d.io.read_point_cloud(str(pcd_path))

                        # Subsample points for visualization (for performance)
                        if len(pcd.points) > 0:
                            st.write("## 3D Point Cloud Visualization")
                            # Downsample more aggressively for larger point clouds
                            sample_rate = max(1, len(pcd.points) // 10000)  # Target ~10k points for viz
                            vis_points = np.asarray(pcd.points)[::sample_rate]

                            # Handle colors if available
                            if pcd.has_colors():
                                vis_colors = np.asarray(pcd.colors)[::sample_rate]
                                # Convert RGB to hex colors for plotly
                                colors = ['rgb({},{},{})'.format(int(r*255), int(g*255), int(b*255))
                                        for r, g, b in vis_colors]
                            else:
                                # Default color if no colors available
                                colors = 'rgb(100,100,100)'

                            # Create plotly figure
                            fig = go.Figure(data=[go.Scatter3d(
                                x=vis_points[:, 0],
                                y=vis_points[:, 1],
                                z=vis_points[:, 2],
                                mode='markers',
                                marker=dict(
                                    size=2,
                                    color=colors,
                                    opacity=0.8
                                )
                            )])

                            fig.update_layout(
                                title=f"Point Cloud Preview ({len(vis_points):,} of {len(pcd.points):,} points)",
                                margin=dict(l=0, r=0, b=0, t=40),
                                scene=dict(
                                    aspectmode='data'
                                )
                            )

                            st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not create 3D visualization: {str(e)}")
            else:
                st.error("Reconstruction failed.")
                # Show error details
                with open(log_path, "r") as f:
                    st.expander("Complete Error Details").code(f.read(), language="text")

st.subheader("Scan History")
for row in get_all_scans():
    st.markdown(f"📁 `{row[0]}` — {row[1]} ({row[2]})")
