# BoxMOT Integration Guide

This guide explains how to use the BoxMOT (Box Multi-Object Tracking) library integrated into the ultralytics_ros package.

## Overview

BoxMOT provides advanced multi-object tracking algorithms that can be used as an alternative to the built-in Ultralytics trackers. The integration supports the following trackers:

- **ByteTrack**: Fast and efficient tracker based on simple association
- **BoTSORT**: Combines motion and appearance information with robust association
- **DeepOCSORT**: Observation-centric SORT with deep appearance features (requires ReID model)
- **OCSORT**: Observation-centric SORT
- **StrongSORT**: Strong appearance-based tracker (requires ReID model)
- **HybridSORT**: Hybrid tracking combining multiple strategies

## Installation

The BoxMOT library is already included in the requirements.txt file and will be installed automatically when you set up the package:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Using BoxMOT with Standard Tracking

To use BoxMOT trackers, launch the tracker node with the BoxMOT launch file:

```bash
ros2 launch ultralytics_ros tracker_boxmot.launch.xml
```

### 2. Using BoxMOT with 3D Point Cloud Tracking

To combine BoxMOT tracking with 3D point cloud data:

```bash
ros2 launch ultralytics_ros tracker_with_cloud_boxmot.launch.xml
```

## Configuration Parameters

### Basic BoxMOT Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tracking_method` | string | `"boxmot"` | Set to "boxmot" to use BoxMOT, or "ultralytics" for built-in trackers |
| `boxmot_tracker` | string | `"deepocsort"` | Tracker algorithm: bytetrack, botsort, deepocsort, ocsort, strongsort, hybridsort |

### ReID Model Parameters (for DeepOCSORT and StrongSORT)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `reid_model` | string | `"osnet_x0_25_msmt17.pt"` | ReID model for appearance-based tracking |

### Tracking Thresholds

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `track_high_thresh` | double | `0.5` | High confidence threshold for track initialization |
| `track_low_thresh` | double | `0.1` | Low confidence threshold for track maintenance |
| `new_track_thresh` | double | `0.6` | Threshold for creating new tracks |
| `track_buffer` | int | `30` | Number of frames to keep lost tracks |
| `match_thresh` | double | `0.8` | Matching threshold for association |

## Examples

### Example 1: Using ByteTrack (Fastest)

```bash
ros2 launch ultralytics_ros tracker_boxmot.launch.xml \
    boxmot_tracker:=bytetrack \
    input_topic:=/camera/image_raw
```

### Example 2: Using DeepOCSORT (Best Accuracy)

```bash
ros2 launch ultralytics_ros tracker_boxmot.launch.xml \
    boxmot_tracker:=deepocsort \
    reid_model:=osnet_x0_25_msmt17.pt \
    track_high_thresh:=0.6 \
    new_track_thresh:=0.7 \
    input_topic:=/camera/image_raw
```

### Example 3: Using BoTSORT with Custom Parameters

```bash
ros2 launch ultralytics_ros tracker_boxmot.launch.xml \
    boxmot_tracker:=botsort \
    track_buffer:=50 \
    match_thresh:=0.9 \
    device:=cuda:0
```

### Example 4: Switching Back to Ultralytics Built-in Tracker

```bash
ros2 launch ultralytics_ros tracker.launch.xml \
    tracker:=bytetrack.yaml
```

Or modify the tracking_method parameter:

```bash
ros2 launch ultralytics_ros tracker_boxmot.launch.xml \
    tracking_method:=ultralytics \
    tracker:=bytetrack.yaml
```

## Tracker Comparison

| Tracker | Speed | Accuracy | ReID Required | Best For |
|---------|-------|----------|---------------|----------|
| ByteTrack | ⚡⚡⚡ | ⭐⭐ | ❌ | Real-time applications, high FPS |
| OCSORT | ⚡⚡ | ⭐⭐⭐ | ❌ | Balanced speed and accuracy |
| BoTSORT | ⚡⚡ | ⭐⭐⭐⭐ | ❌ | Good balance, occlusion handling |
| DeepOCSORT | ⚡ | ⭐⭐⭐⭐⭐ | ✅ | Best accuracy, appearance features |
| StrongSORT | ⚡ | ⭐⭐⭐⭐⭐ | ✅ | Crowded scenes, long occlusions |
| HybridSORT | ⚡⚡ | ⭐⭐⭐⭐ | ❌ | Hybrid approach |

## ReID Models

For DeepOCSORT and StrongSORT, you need to download ReID models. BoxMOT will automatically download the default model `osnet_x0_25_msmt17.pt` on first use.

Available ReID models:
- `osnet_x0_25_msmt17.pt` (lightweight, fast)
- `osnet_x0_5_msmt17.pt` (balanced)
- `osnet_x1_0_msmt17.pt` (best accuracy)

## Troubleshooting

### Issue: ModuleNotFoundError: No module named 'boxmot'

**Solution**: Install the dependencies:
```bash
cd /path/to/ultralytics_ros1
pip install -r requirements.txt
```

### Issue: Tracker runs slowly with DeepOCSORT/StrongSORT

**Solution**: These trackers use ReID models which require GPU for best performance. Set the device parameter:
```bash
ros2 launch ultralytics_ros tracker_boxmot.launch.xml device:=cuda:0
```

### Issue: Tracking quality is poor

**Solution**: Try adjusting the tracking thresholds:
- Increase `track_high_thresh` and `new_track_thresh` for more stable tracks
- Increase `track_buffer` to maintain tracks longer during occlusions
- Increase `match_thresh` for stricter matching

### Issue: Want to use Ultralytics built-in trackers

**Solution**: Either use the original launch file or set `tracking_method:=ultralytics`:
```bash
ros2 launch ultralytics_ros tracker.launch.xml
# OR
ros2 launch ultralytics_ros tracker_boxmot.launch.xml tracking_method:=ultralytics
```

## Performance Tips

1. **For real-time performance**: Use ByteTrack or OCSORT with CPU/GPU acceleration
2. **For best accuracy**: Use DeepOCSORT or StrongSORT with GPU
3. **For crowded scenes**: Use StrongSORT or DeepOCSORT with ReID
4. **For fast-moving objects**: Increase `track_buffer` to maintain tracks during brief occlusions
5. **To reduce false positives**: Increase `new_track_thresh` and `track_high_thresh`

## References

- BoxMOT GitHub: https://github.com/mikel-brostrom/boxmot
- Ultralytics Tracking: https://docs.ultralytics.com/modes/track/
- ByteTrack Paper: https://arxiv.org/abs/2110.06864
- OC-SORT Paper: https://arxiv.org/abs/2203.14360
- StrongSORT Paper: https://arxiv.org/abs/2202.13514
