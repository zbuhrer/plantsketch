# app.py
import streamlit as st
from pathlib import Path
import shutil

from db import init_db, insert_scan, get_all_scans
from extract_frames import extract_frames
from reconstruction import create_project_dir, run_meshroom

init_db()
st.title("Plantsketch")

video_file = st.file_uploader("Upload a garden video", type=["mp4", "mov", "avi"])
fps = st.slider("FPS to extract", 1, 10, 2)

if video_file:
    st.success("Video received.")
    if st.button("Start Scan"):
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

st.subheader("Scan History")
for row in get_all_scans():
    st.markdown(f"📁 `{row[0]}` — {row[1]} ({row[2]})")
