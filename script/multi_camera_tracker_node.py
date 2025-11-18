#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ultralytics_ros
# Copyright (C) 2023-2024  Alpaca-zip
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import cv_bridge
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from sensor_msgs.msg import Image
from ultralytics import YOLO
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
from ultralytics_ros.msg import YoloResult
from boxmot import DeepOCSORT, BoTSORT, StrongSORT
from message_filters import Subscriber, ApproximateTimeSynchronizer
import threading


class GlobalTracker:
    """
    Global tracker for managing tracks across multiple cameras.
    Handles track ID assignment and maintains consistent IDs across camera views.
    """
    def __init__(self):
        self.global_tracks = {}  # {global_id: {camera_id: local_id, last_seen: timestamp, bbox: [x,y,w,h,angle], class: str}}
        self.next_global_id = 1
        self.lock = threading.Lock()
        self.id_mapping = {}  # {(camera_id, local_id): global_id}

    def update_tracks(self, camera_id, tracks, class_names, timestamp):
        """
        Update global tracks with detections from a specific camera.

        Args:
            camera_id: Camera identifier (0-3 for 4-camera setup)
            tracks: BoxMOT tracks array (x1, y1, x2, y2, track_id, conf, cls, ...)
            class_names: Dictionary mapping class IDs to names
            timestamp: Current timestamp

        Returns:
            Mapping from local track IDs to global track IDs
        """
        with self.lock:
            local_to_global = {}

            if tracks is None or len(tracks) == 0:
                return local_to_global

            for track in tracks:
                x1, y1, x2, y2, local_id, conf, cls = track[:7]
                local_id = int(local_id)
                cls_name = class_names.get(int(cls), str(int(cls)))

                # Calculate center and dimensions
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                w = x2 - x1
                h = y2 - y1

                # Calculate angle based on camera position (each camera covers 90°)
                # Camera 0: 0-90°, Camera 1: 90-180°, Camera 2: 180-270°, Camera 3: 270-360°
                angle_offset = camera_id * 90

                # Check if this local track is already mapped to a global ID
                key = (camera_id, local_id)
                if key in self.id_mapping:
                    global_id = self.id_mapping[key]
                    # Update existing track
                    if global_id in self.global_tracks:
                        self.global_tracks[global_id]['last_seen'] = timestamp
                        self.global_tracks[global_id]['camera_id'] = camera_id
                        self.global_tracks[global_id]['bbox'] = [cx, cy, w, h, angle_offset]
                        self.global_tracks[global_id]['conf'] = conf
                    local_to_global[local_id] = global_id
                else:
                    # Try to match with existing global tracks from adjacent cameras
                    matched_global_id = self._match_adjacent_camera(
                        camera_id, cls_name, cx, cy, w, h, timestamp
                    )

                    if matched_global_id is not None:
                        # Matched with existing global track
                        global_id = matched_global_id
                        self.id_mapping[key] = global_id
                        self.global_tracks[global_id]['last_seen'] = timestamp
                        self.global_tracks[global_id]['camera_id'] = camera_id
                        self.global_tracks[global_id]['bbox'] = [cx, cy, w, h, angle_offset]
                    else:
                        # Create new global track
                        global_id = self.next_global_id
                        self.next_global_id += 1
                        self.id_mapping[key] = global_id
                        self.global_tracks[global_id] = {
                            'camera_id': camera_id,
                            'last_seen': timestamp,
                            'bbox': [cx, cy, w, h, angle_offset],
                            'class': cls_name,
                            'conf': conf
                        }

                    local_to_global[local_id] = global_id

            # Clean up old tracks (not seen for more than 2 seconds)
            self._cleanup_old_tracks(timestamp, timeout=2.0)

            return local_to_global

    def _match_adjacent_camera(self, camera_id, cls_name, cx, cy, w, h, timestamp, threshold=0.3):
        """
        Try to match a detection with global tracks from adjacent cameras.
        This handles objects transitioning between camera views.
        """
        # Get adjacent camera IDs (in 360° setup, cameras are arranged in a circle)
        adjacent_cameras = [(camera_id - 1) % 4, (camera_id + 1) % 4]

        best_match = None
        best_score = threshold

        for global_id, track_info in self.global_tracks.items():
            # Only consider tracks from adjacent cameras
            if track_info['camera_id'] not in adjacent_cameras:
                continue

            # Only match same class
            if track_info['class'] != cls_name:
                continue

            # Check if track is recent (within 0.5 seconds)
            if timestamp - track_info['last_seen'] > 0.5:
                continue

            # Calculate IoU-like similarity (simplified for cross-camera matching)
            track_cx, track_cy, track_w, track_h, _ = track_info['bbox']

            # For edge objects, they should appear at opposite edges of adjacent cameras
            # This is a simplified heuristic - you may need to calibrate this for your setup
            size_similarity = min(w, track_w) / max(w, track_w) * min(h, track_h) / max(h, track_h)

            # Vertical position should be similar
            vertical_diff = abs(cy - track_cy) / max(cy, track_cy) if max(cy, track_cy) > 0 else 0

            score = size_similarity * (1 - vertical_diff)

            if score > best_score:
                best_score = score
                best_match = global_id

        return best_match

    def _cleanup_old_tracks(self, current_time, timeout=2.0):
        """Remove tracks that haven't been seen for a while."""
        to_remove = []
        for global_id, track_info in self.global_tracks.items():
            if current_time - track_info['last_seen'] > timeout:
                to_remove.append(global_id)

        for global_id in to_remove:
            # Remove from global tracks
            del self.global_tracks[global_id]
            # Remove from ID mapping
            keys_to_remove = [k for k, v in self.id_mapping.items() if v == global_id]
            for key in keys_to_remove:
                del self.id_mapping[key]


class MultiCameraTrackerNode(Node):
    """
    Multi-camera tracker node for 360° panoramic tracking.
    Handles 4 cameras with BoxMOT tracking and global ID management.
    """
    def __init__(self):
        super().__init__("multi_camera_tracker_node")

        # Declare parameters
        self.declare_parameter("yolo_model", "yolov8n.pt")
        self.declare_parameter("num_cameras", 4)
        self.declare_parameter("input_topics", ["/camera0/image_raw", "/camera1/image_raw",
                                                 "/camera2/image_raw", "/camera3/image_raw"])
        self.declare_parameter("result_topic_prefix", "/yolo_result")
        self.declare_parameter("result_image_topic_prefix", "/yolo_image")
        self.declare_parameter("conf_thres", 0.25)
        self.declare_parameter("iou_thres", 0.45)
        self.declare_parameter("max_det", 300)
        self.declare_parameter("classes", list(range(80)))
        self.declare_parameter("device", "cpu")
        self.declare_parameter("result_conf", True)
        self.declare_parameter("result_line_width", 1)
        self.declare_parameter("result_font_size", 1)
        self.declare_parameter("result_font", "Arial.ttf")
        self.declare_parameter("result_labels", True)
        self.declare_parameter("result_boxes", True)

        # BoxMOT parameters
        self.declare_parameter("boxmot_tracker", "deepocsort")
        self.declare_parameter("reid_model", "osnet_x0_25_msmt17.pt")
        self.declare_parameter("track_high_thresh", 0.5)
        self.declare_parameter("track_low_thresh", 0.1)
        self.declare_parameter("new_track_thresh", 0.6)
        self.declare_parameter("track_buffer", 30)
        self.declare_parameter("match_thresh", 0.8)

        # Multi-camera parameters
        self.declare_parameter("use_time_sync", True)  # Synchronize camera inputs
        self.declare_parameter("sync_queue_size", 10)
        self.declare_parameter("sync_slop", 0.1)  # 100ms tolerance for synchronization

        # Initialize YOLO model
        path = get_package_share_directory("ultralytics_ros")
        yolo_model = self.get_parameter("yolo_model").get_parameter_value().string_value
        self.model = YOLO(f"{path}/models/{yolo_model}")
        self.model.fuse()

        self.bridge = cv_bridge.CvBridge()
        self.use_segmentation = yolo_model.endswith("-seg.pt")

        # Initialize global tracker
        self.global_tracker = GlobalTracker()

        # Initialize BoxMOT trackers for each camera
        self.num_cameras = self.get_parameter("num_cameras").get_parameter_value().integer_value
        self.trackers = [self._create_boxmot_tracker() for _ in range(self.num_cameras)]

        # Get input topics
        input_topics = self.get_parameter("input_topics").get_parameter_value().string_array_value
        if len(input_topics) != self.num_cameras:
            self.get_logger().error(f"Number of input topics ({len(input_topics)}) doesn't match num_cameras ({self.num_cameras})")
            return

        # Setup publishers for each camera
        result_topic_prefix = self.get_parameter("result_topic_prefix").get_parameter_value().string_value
        result_image_topic_prefix = self.get_parameter("result_image_topic_prefix").get_parameter_value().string_value

        self.results_pubs = []
        self.result_image_pubs = []
        for i in range(self.num_cameras):
            self.results_pubs.append(
                self.create_publisher(YoloResult, f"{result_topic_prefix}_camera{i}", 1)
            )
            self.result_image_pubs.append(
                self.create_publisher(Image, f"{result_image_topic_prefix}_camera{i}", 1)
            )

        # Setup subscribers
        use_time_sync = self.get_parameter("use_time_sync").get_parameter_value().bool_value

        if use_time_sync and self.num_cameras > 1:
            # Use time synchronization for multiple cameras
            self.get_logger().info("Using time-synchronized multi-camera input")
            self.image_subs = [Subscriber(self, Image, topic) for topic in input_topics]
            sync_queue_size = self.get_parameter("sync_queue_size").get_parameter_value().integer_value
            sync_slop = self.get_parameter("sync_slop").get_parameter_value().double_value

            self.sync = ApproximateTimeSynchronizer(
                self.image_subs, sync_queue_size, sync_slop
            )
            self.sync.registerCallback(self.synchronized_image_callback)
        else:
            # Process each camera independently
            self.get_logger().info("Using independent processing for each camera")
            for i, topic in enumerate(input_topics):
                self.create_subscription(
                    Image, topic,
                    lambda msg, cam_id=i: self.single_image_callback(msg, cam_id),
                    1
                )

        self.get_logger().info(f"Multi-camera tracker initialized with {self.num_cameras} cameras")

    def _create_boxmot_tracker(self):
        """Create a BoxMOT tracker instance."""
        tracker_type = self.get_parameter("boxmot_tracker").get_parameter_value().string_value
        device = self.get_parameter("device").get_parameter_value().string_value
        track_high_thresh = self.get_parameter("track_high_thresh").get_parameter_value().double_value
        track_low_thresh = self.get_parameter("track_low_thresh").get_parameter_value().double_value
        new_track_thresh = self.get_parameter("new_track_thresh").get_parameter_value().double_value
        track_buffer = self.get_parameter("track_buffer").get_parameter_value().integer_value
        match_thresh = self.get_parameter("match_thresh").get_parameter_value().double_value

        tracker_args = {
            "track_high_thresh": track_high_thresh,
            "track_low_thresh": track_low_thresh,
            "new_track_thresh": new_track_thresh,
            "track_buffer": track_buffer,
            "match_thresh": match_thresh,
            "frame_rate": 30,
        }

        if tracker_type == "deepocsort":
            reid_model = self.get_parameter("reid_model").get_parameter_value().string_value
            return DeepOCSORT(model_weights=reid_model, device=device, **tracker_args)
        elif tracker_type == "botsort":
            return BoTSORT(**tracker_args)
        elif tracker_type == "strongsort":
            reid_model = self.get_parameter("reid_model").get_parameter_value().string_value
            return StrongSORT(model_weights=reid_model, device=device, **tracker_args)
        else:
            self.get_logger().warn(f"Unknown tracker type: {tracker_type}, using DeepOCSORT")
            reid_model = self.get_parameter("reid_model").get_parameter_value().string_value
            return DeepOCSORT(model_weights=reid_model, device=device, **tracker_args)

    def synchronized_image_callback(self, *msgs):
        """Callback for synchronized multi-camera images."""
        timestamp = self.get_clock().now().nanoseconds / 1e9

        for camera_id, msg in enumerate(msgs):
            self.process_camera_image(msg, camera_id, timestamp)

    def single_image_callback(self, msg, camera_id):
        """Callback for single camera image (non-synchronized mode)."""
        timestamp = self.get_clock().now().nanoseconds / 1e9
        self.process_camera_image(msg, camera_id, timestamp)

    def process_camera_image(self, msg, camera_id, timestamp):
        """Process image from a specific camera."""
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        # Get parameters
        conf_thres = self.get_parameter("conf_thres").get_parameter_value().double_value
        iou_thres = self.get_parameter("iou_thres").get_parameter_value().double_value
        max_det = self.get_parameter("max_det").get_parameter_value().integer_value
        classes = self.get_parameter("classes").get_parameter_value().integer_array_value
        device = self.get_parameter("device").get_parameter_value().string_value or None

        # Run YOLO detection
        results = self.model(
            source=cv_image,
            conf=conf_thres,
            iou=iou_thres,
            max_det=max_det,
            classes=classes,
            device=device,
            verbose=False,
            retina_masks=True,
        )

        # Apply BoxMOT tracking
        if results is not None and len(results) > 0 and results[0].boxes is not None:
            dets = results[0].boxes.data.cpu().numpy()
            if len(dets) > 0:
                # Apply local tracking
                tracks = self.trackers[camera_id].update(dets, cv_image)

                if tracks is not None and len(tracks) > 0:
                    # Update global tracker and get global IDs
                    class_names = results[0].names
                    local_to_global = self.global_tracker.update_tracks(
                        camera_id, tracks, class_names, timestamp
                    )

                    # Map local IDs to global IDs
                    global_ids = []
                    for track in tracks:
                        local_id = int(track[4])
                        global_id = local_to_global.get(local_id, local_id)
                        global_ids.append(global_id)

                    # Update results with global track IDs
                    results[0].boxes.id = np.array(global_ids)

        # Publish results
        if results is not None:
            yolo_result_msg = YoloResult()
            yolo_result_msg.header = msg.header
            yolo_result_msg.header.frame_id = f"camera{camera_id}"
            yolo_result_msg.detections = self.create_detections_array(results)

            yolo_result_image_msg = self.create_result_image(results)
            yolo_result_image_msg.header = msg.header
            yolo_result_image_msg.header.frame_id = f"camera{camera_id}"

            if self.use_segmentation:
                yolo_result_msg.masks = self.create_segmentation_masks(results)

            self.results_pubs[camera_id].publish(yolo_result_msg)
            self.result_image_pubs[camera_id].publish(yolo_result_image_msg)

    def create_detections_array(self, results):
        """Create Detection2DArray from YOLO results."""
        detections_msg = Detection2DArray()

        if results[0].boxes is None or len(results[0].boxes) == 0:
            return detections_msg

        bounding_box = results[0].boxes.xywh
        classes = results[0].boxes.cls
        confidence_score = results[0].boxes.conf
        track_ids = results[0].boxes.id if results[0].boxes.id is not None else [None] * len(bounding_box)

        for bbox, cls, conf, track_id in zip(bounding_box, classes, confidence_score, track_ids):
            detection = Detection2D()
            detection.bbox.center.position.x = float(bbox[0])
            detection.bbox.center.position.y = float(bbox[1])
            detection.bbox.size_x = float(bbox[2])
            detection.bbox.size_y = float(bbox[3])

            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = results[0].names.get(int(cls))
            hypothesis.hypothesis.score = float(conf)

            detection.results.append(hypothesis)

            # Add track ID as part of the detection (stored in detection.id if available in your msg definition)
            # For now, it's included in the class_id with format "classname_id"
            if track_id is not None:
                hypothesis.hypothesis.class_id = f"{results[0].names.get(int(cls))}_id{int(track_id)}"

            detections_msg.detections.append(detection)

        return detections_msg

    def create_result_image(self, results):
        """Create annotated result image."""
        result_conf = self.get_parameter("result_conf").get_parameter_value().bool_value
        result_line_width = self.get_parameter("result_line_width").get_parameter_value().integer_value
        result_font_size = self.get_parameter("result_font_size").get_parameter_value().integer_value
        result_font = self.get_parameter("result_font").get_parameter_value().string_value
        result_labels = self.get_parameter("result_labels").get_parameter_value().bool_value
        result_boxes = self.get_parameter("result_boxes").get_parameter_value().bool_value

        plotted_image = results[0].plot(
            conf=result_conf,
            line_width=result_line_width,
            font_size=result_font_size,
            font=result_font,
            labels=result_labels,
            boxes=result_boxes,
        )

        result_image_msg = self.bridge.cv2_to_imgmsg(plotted_image, encoding="bgr8")
        return result_image_msg

    def create_segmentation_masks(self, results):
        """Create segmentation masks."""
        masks_msg = []
        for result in results:
            if hasattr(result, "masks") and result.masks is not None:
                for mask_tensor in result.masks:
                    mask_numpy = (
                        np.squeeze(mask_tensor.data.to("cpu").detach().numpy()).astype(np.uint8) * 255
                    )
                    mask_image_msg = self.bridge.cv2_to_imgmsg(mask_numpy, encoding="mono8")
                    masks_msg.append(mask_image_msg)
        return masks_msg


def main(args=None):
    rclpy.init(args=args)
    node = MultiCameraTrackerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
