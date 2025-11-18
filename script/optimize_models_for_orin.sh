#!/bin/bash

##############################################################################
# Batch TensorRT Optimization Script for NVIDIA AGX Orin
#
# This script converts multiple YOLO models to TensorRT format optimized
# for AGX Orin deployment.
#
# Usage:
#   ./optimize_models_for_orin.sh [options]
#
# Options:
#   --single-camera    Optimize for single camera (batch=1, 640x640)
#   --multi-camera     Optimize for 4-camera system (batch=1, 1920x1080)
#   --all             Optimize for both configurations
#   --models          Specify models (default: yolov8n yolov8m yolov8l)
##############################################################################

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default settings
MODE="all"
MODELS="yolov8n yolov8m yolov8l"
WORKSPACE=4
DEVICE=0

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --single-camera)
            MODE="single"
            shift
            ;;
        --multi-camera)
            MODE="multi"
            shift
            ;;
        --all)
            MODE="all"
            shift
            ;;
        --models)
            MODELS="$2"
            shift 2
            ;;
        --workspace)
            WORKSPACE="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --single-camera    Optimize for single camera (640x640)"
            echo "  --multi-camera     Optimize for 4-camera system (1920x1080)"
            echo "  --all              Optimize for both configurations (default)"
            echo "  --models          Specify models (default: yolov8n yolov8m yolov8l)"
            echo "  --workspace       TensorRT workspace in GB (default: 4)"
            echo "  --device          CUDA device ID (default: 0)"
            echo "  --help            Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     TensorRT Model Optimization for NVIDIA AGX Orin               ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
MODELS_DIR="$(dirname "$SCRIPT_DIR")/models"

# Create models directory if it doesn't exist
mkdir -p "$MODELS_DIR"

echo -e "${GREEN}Configuration:${NC}"
echo -e "  Mode:       $MODE"
echo -e "  Models:     $MODELS"
echo -e "  Workspace:  ${WORKSPACE}GB"
echo -e "  Device:     cuda:$DEVICE"
echo -e "  Models dir: $MODELS_DIR"
echo ""

# Check if Python script exists
CONVERT_SCRIPT="$SCRIPT_DIR/convert_to_tensorrt.py"
if [ ! -f "$CONVERT_SCRIPT" ]; then
    echo -e "${RED}Error: convert_to_tensorrt.py not found at $CONVERT_SCRIPT${NC}"
    exit 1
fi

# Check CUDA availability
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}Error: nvidia-smi not found. Is CUDA installed?${NC}"
    exit 1
fi

echo -e "${GREEN}NVIDIA GPU Information:${NC}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
echo ""

# Function to convert a model
convert_model() {
    local model_name=$1
    local imgsz=$2
    local batch=$3
    local config_name=$4

    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}Converting: ${model_name} [${config_name}]${NC}"
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    local model_path="$MODELS_DIR/${model_name}.pt"

    # Download model if it doesn't exist
    if [ ! -f "$model_path" ]; then
        echo -e "${BLUE}Downloading $model_name...${NC}"
        python3 -c "from ultralytics import YOLO; YOLO('$model_name.pt').model" || {
            echo -e "${RED}Failed to download $model_name${NC}"
            return 1
        }
        # Move downloaded model to models directory
        if [ -f "${model_name}.pt" ]; then
            mv "${model_name}.pt" "$model_path"
        fi
    fi

    # Convert to TensorRT
    python3 "$CONVERT_SCRIPT" \
        --model "$model_path" \
        --imgsz $imgsz \
        --batch $batch \
        --workspace $WORKSPACE \
        --device $DEVICE \
        --half || {
        echo -e "${RED}Failed to convert $model_name${NC}"
        return 1
    }

    echo -e "${GREEN}✓ Successfully converted ${model_name} [${config_name}]${NC}"
    echo ""
}

# Counter for tracking progress
total_conversions=0
successful_conversions=0

# Convert models based on mode
for model in $MODELS; do
    if [ "$MODE" = "single" ] || [ "$MODE" = "all" ]; then
        ((total_conversions++))
        if convert_model "$model" "640 640" 1 "Single Camera"; then
            ((successful_conversions++))
        fi
    fi

    if [ "$MODE" = "multi" ] || [ "$MODE" = "all" ]; then
        ((total_conversions++))
        if convert_model "$model" "1920 1080" 1 "Multi Camera"; then
            ((successful_conversions++))
        fi
    fi
done

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                    Optimization Summary                           ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}Successful: $successful_conversions / $total_conversions${NC}"
echo ""

# List all generated engines
echo -e "${YELLOW}Generated TensorRT Engines:${NC}"
find "$MODELS_DIR" -name "*.engine" -type f -exec basename {} \; | sort
echo ""

# Show usage instructions
echo -e "${BLUE}╔════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                      Usage Instructions                           ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}To use these engines with ROS2:${NC}"
echo ""
echo -e "1. Single camera (640x640):"
echo -e "   ${YELLOW}ros2 launch ultralytics_ros tracker.launch.xml \\${NC}"
echo -e "   ${YELLOW}       yolo_model:=yolov8m_640x640_fp16_batch1.engine \\${NC}"
echo -e "   ${YELLOW}       device:=cuda:0${NC}"
echo ""
echo -e "2. Multi-camera (1920x1080, 4 cameras):"
echo -e "   ${YELLOW}ros2 launch ultralytics_ros multi_camera_tracker.launch.xml \\${NC}"
echo -e "   ${YELLOW}       yolo_model:=yolov8m_1920x1080_fp16_batch1.engine \\${NC}"
echo -e "   ${YELLOW}       device:=cuda:0${NC}"
echo ""
echo -e "${GREEN}Performance Benchmarking:${NC}"
echo ""
echo -e "   ${YELLOW}python3 $CONVERT_SCRIPT \\${NC}"
echo -e "   ${YELLOW}       --model $MODELS_DIR/yolov8m_640x640_fp16_batch1.engine \\${NC}"
echo -e "   ${YELLOW}       --benchmark${NC}"
echo ""

if [ $successful_conversions -eq $total_conversions ]; then
    echo -e "${GREEN}✓ All conversions completed successfully!${NC}"
    exit 0
else
    echo -e "${YELLOW}⚠ Some conversions failed. Check the output above for details.${NC}"
    exit 1
fi
