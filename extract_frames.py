# extract_frames.py
import cv2
import os
from pathlib import Path

def extract_frames(video_path, output_dir, fps=2):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("Cannot open video")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    saved = 0
    success = True

    while success:
        success, frame = cap.read()
        if not success:
            break

        frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        frame_rate = cap.get(cv2.CAP_PROP_FPS)

        if frame_rate == 0 or frame_number % int(frame_rate // fps + 1) == 0:
            out_path = Path(output_dir) / f"frame_{saved:04d}.jpg"
            cv2.imwrite(str(out_path), frame)
            saved += 1

    cap.release()
    return saved
