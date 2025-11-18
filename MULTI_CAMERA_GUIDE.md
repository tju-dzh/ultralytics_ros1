# Multi-Camera 360° Panoramic Tracking Guide

This guide explains how to use the multi-camera tracking system for 360° panoramic object detection and tracking with the ultralytics_ros package.

## Overview

The multi-camera tracking system is designed for panoramic camera setups, such as a 4-camera system with 360°×80° field of view. It provides:

- **Simultaneous multi-camera processing**: Process images from 4 cameras simultaneously
- **Global track ID management**: Consistent object IDs across all camera views
- **Cross-camera tracking**: Objects transitioning between camera views maintain their IDs
- **BoxMOT integration**: Advanced tracking with DeepOCSORT, BoTSORT, or StrongSORT
- **Time synchronization**: Optional timestamp-based synchronization for all cameras
- **3D tracking support**: Integration with LiDAR for distance measurement

## System Architecture

### Camera Setup (360° Coverage)

```
                    Front (Camera 0)
                         0°-90°
                           |
    Left (Camera 3)  ------+------  Right (Camera 1)
    270°-360°              |              90°-180°
                           |
                    Back (Camera 2)
                        180°-270°
```

Each camera covers approximately 90° horizontally, providing complete 360° coverage.

## Hardware Requirements

### Recommended Configuration

- **Cameras**: 4× 1080P cameras with synchronized timestamps
- **Field of View**: 360°×80° total coverage (90° per camera horizontally)
- **Frame Rate**: 30 FPS recommended
- **GPU**: NVIDIA GPU with CUDA support (for real-time performance with ReID models)
- **RAM**: 8GB+ recommended
- **CPU**: Multi-core processor for parallel processing

### Camera Calibration

Ensure your cameras are:
1. Properly calibrated (camera_info topics available)
2. Time-synchronized (hardware sync recommended)
3. Mounted at the same height and orientation
4. Overlapping fields of view at boundaries (optional but helpful)

## Installation

The multi-camera tracker is included in the package. Ensure BoxMOT is installed:

```bash
cd /path/to/ultralytics_ros1
pip install -r requirements.txt
```

## Usage

### Basic Multi-Camera Tracking

Launch the multi-camera tracker with default settings:

```bash
ros2 launch ultralytics_ros multi_camera_tracker.launch.xml
```

### Custom Camera Topics

Specify your camera topics:

```bash
ros2 launch ultralytics_ros multi_camera_tracker.launch.xml \
    camera0_topic:=/front_camera/image_raw \
    camera1_topic:=/right_camera/image_raw \
    camera2_topic:=/back_camera/image_raw \
    camera3_topic:=/left_camera/image_raw \
    device:=cuda:0
```

### With 3D Point Cloud Tracking

Combine multi-camera tracking with LiDAR for distance measurement:

```bash
ros2 launch ultralytics_ros multi_camera_tracker_with_cloud.launch.xml \
    camera0_topic:=/camera0/image_raw \
    camera1_topic:=/camera1/image_raw \
    camera2_topic:=/camera2/image_raw \
    camera3_topic:=/camera3/image_raw \
    camera0_info_topic:=/camera0/camera_info \
    camera1_info_topic:=/camera1/camera_info \
    camera2_info_topic:=/camera2/camera_info \
    camera3_info_topic:=/camera3/camera_info \
    lidar_topic:=/lidar/points \
    device:=cuda:0
```

### Advanced Configuration

#### High-Performance Setup (GPU-accelerated)

```bash
ros2 launch ultralytics_ros multi_camera_tracker.launch.xml \
    yolo_model:=yolov8l.pt \
    boxmot_tracker:=deepocsort \
    reid_model:=osnet_x1_0_msmt17.pt \
    device:=cuda:0 \
    conf_thres:=0.3 \
    track_high_thresh:=0.6 \
    new_track_thresh:=0.7 \
    use_time_sync:=true
```

#### CPU-Only Setup

```bash
ros2 launch ultralytics_ros multi_camera_tracker.launch.xml \
    yolo_model:=yolov8n.pt \
    boxmot_tracker:=botsort \
    device:=cpu \
    use_time_sync:=false
```

## Configuration Parameters

### Camera Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_cameras` | int | `4` | Number of cameras in the system |
| `camera0_topic` | string | `/camera0/image_raw` | Topic for camera 0 (front) |
| `camera1_topic` | string | `/camera1/image_raw` | Topic for camera 1 (right) |
| `camera2_topic` | string | `/camera2/image_raw` | Topic for camera 2 (back) |
| `camera3_topic` | string | `/camera3/image_raw` | Topic for camera 3 (left) |

### Synchronization Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_time_sync` | bool | `true` | Enable time-synchronized processing of all cameras |
| `sync_queue_size` | int | `10` | Queue size for message synchronizer |
| `sync_slop` | double | `0.1` | Maximum time difference (seconds) between messages |

**Note**:
- Set `use_time_sync=true` for hardware-synchronized cameras
- Set `use_time_sync=false` for independent camera processing (higher throughput but no sync)

### Output Topics

Each camera produces separate output topics:

- `/yolo_result_camera0` - Detection results from camera 0
- `/yolo_result_camera1` - Detection results from camera 1
- `/yolo_result_camera2` - Detection results from camera 2
- `/yolo_result_camera3` - Detection results from camera 3

Similarly for image outputs:
- `/yolo_image_camera0`, `/yolo_image_camera1`, etc.

For 3D tracking:
- `/yolo_3d_result_camera0`, `/yolo_3d_result_camera1`, etc.

## Global Track ID Management

### How It Works

The multi-camera tracker maintains global track IDs that persist across camera views:

1. **Local Tracking**: Each camera runs its own BoxMOT tracker
2. **Global ID Assignment**: A global tracker assigns unique IDs across all cameras
3. **Cross-Camera Matching**: Objects moving between cameras are matched based on:
   - Object class
   - Bounding box size
   - Temporal proximity
   - Camera adjacency

### Example Scenario

```
Camera 0 (Front):    ID=5 (person)
                         ↓ moves right
Camera 1 (Right):    ID=5 (same person, maintained ID)
                         ↓ moves backward
Camera 2 (Back):     ID=5 (same person, maintained ID)
```

### Tuning Cross-Camera Tracking

To improve cross-camera tracking:

1. **Increase track_buffer**: Maintains tracks longer during transitions
   ```bash
   track_buffer:=50
   ```

2. **Adjust matching thresholds**: Lower thresholds for more aggressive matching
   ```bash
   match_thresh:=0.7
   ```

3. **Use ReID models**: DeepOCSORT or StrongSORT for appearance-based matching
   ```bash
   boxmot_tracker:=deepocsort
   reid_model:=osnet_x1_0_msmt17.pt
   ```

## Performance Optimization

### For 1080P @ 30 FPS (4 cameras simultaneously)

**Recommended Hardware**:
- GPU: NVIDIA RTX 3060 or better
- Model: YOLOv8m or YOLOv8l
- Tracker: DeepOCSORT with osnet_x0_25

**Configuration**:
```bash
ros2 launch ultralytics_ros multi_camera_tracker.launch.xml \
    yolo_model:=yolov8m.pt \
    boxmot_tracker:=deepocsort \
    reid_model:=osnet_x0_25_msmt17.pt \
    device:=cuda:0 \
    use_time_sync:=true \
    sync_slop:=0.05
```

### Performance Tips

1. **Model Selection**:
   - YOLOv8n: Fastest, lower accuracy (~100 FPS per camera on RTX 3060)
   - YOLOv8m: Balanced (~50 FPS per camera)
   - YOLOv8l: Best accuracy (~30 FPS per camera)

2. **Tracker Selection**:
   - BoTSORT: Fastest, no ReID
   - DeepOCSORT: Balanced, with ReID
   - StrongSORT: Most accurate, slowest

3. **GPU Utilization**:
   - Enable CUDA: `device:=cuda:0`
   - Batch processing is automatic across cameras

4. **Synchronization**:
   - Disable if cameras are not hardware-synced: `use_time_sync:=false`
   - Increases throughput but may cause temporal inconsistencies

## Monitoring and Debugging

### Check Topic Output

```bash
# List all topics
ros2 topic list | grep yolo

# Monitor detection rate
ros2 topic hz /yolo_result_camera0

# View results
ros2 run image_view image_view image:=/yolo_image_camera0
```

### Enable Debug Mode

```bash
ros2 launch ultralytics_ros multi_camera_tracker.launch.xml debug:=true
```

This will open image viewers for all 4 cameras.

### Common Issues

#### Issue: High latency or dropped frames

**Solutions**:
1. Reduce model size (use yolov8n instead of yolov8l)
2. Disable time synchronization if not needed
3. Reduce input resolution
4. Use faster tracker (BoTSORT instead of DeepOCSORT)

#### Issue: Lost tracks at camera boundaries

**Solutions**:
1. Increase `track_buffer:=50`
2. Use ReID-based tracker (DeepOCSORT or StrongSORT)
3. Ensure cameras have overlapping fields of view
4. Check camera synchronization

#### Issue: Inconsistent global IDs

**Solutions**:
1. Enable time synchronization: `use_time_sync:=true`
2. Improve camera calibration
3. Use stronger ReID model: `reid_model:=osnet_x1_0_msmt17.pt`
4. Adjust matching threshold: `match_thresh:=0.7`

## Camera Arrangement Best Practices

### Mounting

1. **Height**: Mount all cameras at the same height
2. **Tilt**: Slight downward tilt (10-15°) recommended for ground-level tracking
3. **Overlap**: 5-10° overlap between adjacent cameras helps with cross-camera tracking
4. **Synchronization**: Use hardware trigger for simultaneous capture

### Calibration

1. **Intrinsics**: Calibrate each camera individually
2. **Extrinsics**: Calibrate relative positions if using 3D tracking
3. **Time Sync**: Ensure camera timestamps are synchronized

### Optimal Coverage

```
Front Camera (0): 345° - 75°   (overlap with cameras 1 and 3)
Right Camera (1): 75° - 165°   (overlap with cameras 0 and 2)
Back Camera (2):  165° - 255°  (overlap with cameras 1 and 3)
Left Camera (3):  255° - 345°  (overlap with cameras 2 and 0)
```

## Integration with ROS2 Navigation Stack

The multi-camera tracker can be integrated with ROS2 navigation:

```python
# Subscribe to detection results
ros2 topic echo /yolo_result_camera0

# Use detections for obstacle avoidance
# Each detection includes:
# - bbox: Bounding box (x, y, width, height)
# - class_id: Object class
# - confidence: Detection confidence
# - track_id: Global tracking ID (embedded in class_id as "classname_idN")
```

## Example Application: Autonomous Navigation

```bash
# Terminal 1: Launch multi-camera tracker
ros2 launch ultralytics_ros multi_camera_tracker.launch.xml \
    device:=cuda:0

# Terminal 2: Monitor detections
ros2 topic echo /yolo_result_camera0/detections

# Terminal 3: Visualize all cameras
ros2 run rqt_image_view rqt_image_view
```

## Troubleshooting

### Camera Not Detected

```bash
# Check if camera topics are publishing
ros2 topic list | grep camera

# Check image format
ros2 topic info /camera0/image_raw
```

### Performance Profiling

```bash
# Monitor CPU usage
htop

# Monitor GPU usage
nvidia-smi -l 1

# Check node statistics
ros2 run ros2_introspection ros2_introspection
```

## References

- BoxMOT Documentation: https://github.com/mikel-brostrom/boxmot
- Ultralytics YOLOv8: https://docs.ultralytics.com
- ROS2 message_filters: https://docs.ros.org/en/humble/p/message_filters/
- Multi-Object Tracking: https://arxiv.org/abs/2203.14360

## Support

For issues specific to multi-camera tracking:
1. Check camera synchronization
2. Verify topic names and message types
3. Test with single camera first
4. Enable debug mode for visualization
5. Check system resources (CPU/GPU/RAM)

For general tracking issues, refer to BOXMOT_GUIDE.md
