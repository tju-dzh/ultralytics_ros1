#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TensorRT Model Conversion Script for NVIDIA AGX Orin

This script converts YOLO models to TensorRT format for optimized inference
on NVIDIA AGX Orin platform.

Usage:
    python3 convert_to_tensorrt.py --model yolov8m.pt --imgsz 1920 1080 --half --batch 4
"""

import argparse
import os
import sys
from pathlib import Path
from ultralytics import YOLO


def convert_to_tensorrt(
    model_path,
    imgsz=(640, 640),
    half=True,
    int8=False,
    batch=1,
    workspace=4,
    device=0,
    simplify=True,
    verbose=True
):
    """
    Convert YOLO model to TensorRT format.

    Args:
        model_path (str): Path to YOLO model (.pt file)
        imgsz (tuple): Input image size (height, width) or (size, size)
        half (bool): Use FP16 precision (recommended for AGX Orin)
        int8 (bool): Use INT8 precision (requires calibration data)
        batch (int): Batch size for inference (1 for single camera, 4 for multi-camera)
        workspace (int): TensorRT workspace size in GB
        device (int): CUDA device ID
        simplify (bool): Simplify ONNX model before conversion
        verbose (bool): Verbose output

    Returns:
        str: Path to converted TensorRT engine file
    """
    print("=" * 80)
    print("YOLO Model to TensorRT Conversion for AGX Orin")
    print("=" * 80)

    # Check if model exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Load YOLO model
    print(f"\n[1/4] Loading YOLO model: {model_path}")
    model = YOLO(model_path)

    # Determine output path
    model_name = Path(model_path).stem
    if half:
        precision = "fp16"
    elif int8:
        precision = "int8"
    else:
        precision = "fp32"

    # Handle image size
    if isinstance(imgsz, int):
        imgsz = (imgsz, imgsz)
    elif isinstance(imgsz, (list, tuple)) and len(imgsz) == 1:
        imgsz = (imgsz[0], imgsz[0])

    output_name = f"{model_name}_{imgsz[0]}x{imgsz[1]}_{precision}_batch{batch}.engine"
    output_dir = Path(model_path).parent
    output_path = output_dir / output_name

    print(f"    Model type: {model.task}")
    print(f"    Input size: {imgsz[0]}x{imgsz[1]}")
    print(f"    Precision: {precision.upper()}")
    print(f"    Batch size: {batch}")
    print(f"    Workspace: {workspace} GB")
    print(f"    Device: cuda:{device}")

    # Export to TensorRT
    print(f"\n[2/4] Exporting to TensorRT engine...")
    print(f"    This may take several minutes...")

    try:
        export_path = model.export(
            format="engine",
            imgsz=imgsz,
            half=half,
            int8=int8,
            batch=batch,
            workspace=workspace,
            device=device,
            simplify=simplify,
            verbose=verbose,
        )

        print(f"\n[3/4] TensorRT engine created successfully!")
        print(f"    Engine path: {export_path}")

        # Get file size
        file_size_mb = os.path.getsize(export_path) / (1024 * 1024)
        print(f"    Engine size: {file_size_mb:.2f} MB")

        # Verify engine
        print(f"\n[4/4] Verifying TensorRT engine...")
        test_model = YOLO(export_path)
        print(f"    ✓ Engine loaded successfully")
        print(f"    ✓ Model task: {test_model.task}")

        print("\n" + "=" * 80)
        print("Conversion Complete!")
        print("=" * 80)
        print(f"\nTo use this engine in your ROS2 node:")
        print(f"    yolo_model:={Path(export_path).name}")
        print(f"\nExample launch command:")
        print(f"    ros2 launch ultralytics_ros tracker.launch.xml \\")
        print(f"        yolo_model:={Path(export_path).name} \\")
        print(f"        device:=cuda:0")

        return export_path

    except Exception as e:
        print(f"\n[ERROR] Conversion failed: {e}")
        raise


def benchmark_model(model_path, imgsz=(640, 640), device=0):
    """
    Benchmark TensorRT model performance.

    Args:
        model_path (str): Path to TensorRT engine
        imgsz (tuple): Input image size
        device (int): CUDA device ID
    """
    print("\n" + "=" * 80)
    print("Benchmarking TensorRT Engine")
    print("=" * 80)

    model = YOLO(model_path)

    print(f"\nRunning benchmark on cuda:{device}...")
    print(f"Input size: {imgsz[0]}x{imgsz[1]}")

    # Benchmark using Ultralytics built-in benchmark
    results = model.benchmark(
        imgsz=imgsz,
        device=device,
        verbose=True
    )

    print("\n" + "=" * 80)
    print("Benchmark Complete!")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Convert YOLO models to TensorRT for AGX Orin deployment"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to YOLO model (.pt file)"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        nargs="+",
        default=[640],
        help="Input image size (height width) or (size). For 1080P cameras use: 1920 1080"
    )
    parser.add_argument(
        "--half",
        action="store_true",
        default=True,
        help="Use FP16 precision (default: True, recommended for AGX Orin)"
    )
    parser.add_argument(
        "--no-half",
        action="store_false",
        dest="half",
        help="Disable FP16 precision (use FP32)"
    )
    parser.add_argument(
        "--int8",
        action="store_true",
        help="Use INT8 precision (requires calibration data)"
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help="Batch size (1 for single camera, 4 for 4-camera system)"
    )
    parser.add_argument(
        "--workspace",
        type=int,
        default=4,
        help="TensorRT workspace size in GB (default: 4)"
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="CUDA device ID (default: 0)"
    )
    parser.add_argument(
        "--no-simplify",
        action="store_false",
        dest="simplify",
        help="Disable ONNX model simplification"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmark after conversion"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Verbose output"
    )

    args = parser.parse_args()

    # Parse image size
    if len(args.imgsz) == 1:
        imgsz = (args.imgsz[0], args.imgsz[0])
    elif len(args.imgsz) == 2:
        imgsz = tuple(args.imgsz)
    else:
        raise ValueError("--imgsz must be 1 or 2 values (size) or (height width)")

    # Convert model
    try:
        engine_path = convert_to_tensorrt(
            model_path=args.model,
            imgsz=imgsz,
            half=args.half,
            int8=args.int8,
            batch=args.batch,
            workspace=args.workspace,
            device=args.device,
            simplify=args.simplify,
            verbose=args.verbose
        )

        # Run benchmark if requested
        if args.benchmark:
            benchmark_model(engine_path, imgsz=imgsz, device=args.device)

    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
