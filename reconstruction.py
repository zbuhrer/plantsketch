# reconstruction.py
import os
from pathlib import Path
import subprocess
import uuid

PROJECTS_ROOT = Path("projects")

def create_project_dir():
    uid = str(uuid.uuid4())
    base = PROJECTS_ROOT / uid
    (base / "frames").mkdir(parents=True, exist_ok=True)
    (base / "meshroom_project").mkdir(parents=True, exist_ok=True)
    (base / "output_pointcloud").mkdir(parents=True, exist_ok=True)
    return uid, base

def run_meshroom(frames_dir, output_dir, log_path):
    cmd = [
        "meshroom_batch",
        "--input", str(frames_dir),
        "--output", str(output_dir)
    ]
    with open(log_path, "w") as log_file:
        result = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT)
    return result.returncode == 0
