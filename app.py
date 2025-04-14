# app.py
import streamlit as st
from pathlib import Path
import shutil

from db import init_db, insert_scan, get_all_scans
from extract_frames import extract_frames
from reconstruction import create_project_dir, run_meshroom

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

            video_path = base / "input_video.mp4"
            with open(video_path, "wb") as f:
                f.write(video_file.read())

            st.write("Extracting frames...")
            frame_count = extract_frames(video_path, base / "frames", fps=fps)

            st.write("Running Meshroom...")
            success = run_meshroom(base / "frames", base / "meshroom_project", base / "meshroom_log.txt")

            insert_scan(uuid, video_file.name, frame_count, success)

            if success:
                st.success(f"Scan complete! UUID: {uuid}")
            else:
                st.error("Meshroom failed.")
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
                img_path = frames_dir / f"image_{frame_count:04d}.jpg"
                with open(img_path, "wb") as f:
                    f.write(img.read())
                frame_count += 1

            st.write("Running Meshroom...")
            success = run_meshroom(frames_dir, base / "meshroom_project", base / "meshroom_log.txt")

            # Use a batch name as reference
            reference_name = f"batch_upload_{len(image_files)}_images"

            insert_scan(uuid, reference_name, frame_count, success)

            if success:
                st.success(f"Scan complete! UUID: {uuid}")
            else:
                st.error("Meshroom failed.")

st.subheader("Scan History")
for row in get_all_scans():
    st.markdown(f"📁 `{row[0]}` — {row[1]} ({row[2]})")
