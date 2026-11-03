"""
Download a labeled U.S. coins dataset from Roboflow Universe in YOLOv8 format.

Usage (inside the yolo_gpu environment on the Jetson):
    export ROBOFLOW_API_KEY=your_key_here
    python scripts/download_dataset.py --workspace WORKSPACE --project PROJECT --version 1

Get WORKSPACE / PROJECT / VERSION from the dataset's "Download Dataset" ->
"YOLOv8" -> "Show download code" snippet on universe.roboflow.com.
"""
import argparse
import os
import _path  # noqa: F401

parser = argparse.ArgumentParser()
parser.add_argument("--workspace", required=True)
parser.add_argument("--project", required=True)
parser.add_argument("--version", type=int, default=1)
parser.add_argument("--out", default="data/coins")
args = parser.parse_args()

api_key = os.environ.get("ROBOFLOW_API_KEY")
if not api_key:
    raise SystemExit("Set your key first:  export ROBOFLOW_API_KEY=your_key_here")

from roboflow import Roboflow

rf = Roboflow(api_key=api_key)
project = rf.workspace(args.workspace).project(args.project)
dataset = project.version(args.version).download("yolov8", location=args.out)
print(f"\nDataset saved to: {dataset.location}")
print(f"Train with: python scripts/train.py --data {dataset.location}/data.yaml")
