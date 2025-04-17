# app.py
import streamlit as st
from pathlib import Path
import shutil

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

            st.write("Running 3D reconstruction...")
            log_path = base / "reconstruction_log.txt"
            success = reconstruct_with_open3d(base / "frames", base / "output_pointcloud", log_path)

            insert_scan(uuid, video_file.name, frame_count, success)

            if success:
                st.success(f"Scan complete! UUID: {uuid}")
                # Show the log for verification
                with open(log_path, "r") as f:
                    st.expander("Reconstruction Log").code(f.read(), language="text")
            else:
                st.error("Reconstruction failed.")
                # Show error details
                with open(log_path, "r") as f:
                    st.expander("Error Details").code(f.read(), language="text")
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

            st.write("Running 3D reconstruction...")
            log_path = base / "reconstruction_log.txt"
            success = reconstruct_with_open3d(frames_dir, base / "output_pointcloud", log_path)

            # Use a batch name as reference
            reference_name = f"batch_upload_{len(image_files)}_images"

            insert_scan(uuid, reference_name, frame_count, success)

            if success:
                st.success(f"Scan complete! UUID: {uuid}")
                # Show the log for verification
                with open(log_path, "r") as f:
                    st.expander("Reconstruction Log").code(f.read(), language="text")
            else:
                st.error("Reconstruction failed.")
                # Show error details
                with open(log_path, "r") as f:
                    st.expander("Error Details").code(f.read(), language="text")

st.subheader("Scan History")
for row in get_all_scans():
    st.markdown(f"📁 `{row[0]}` — {row[1]} ({row[2]})")
