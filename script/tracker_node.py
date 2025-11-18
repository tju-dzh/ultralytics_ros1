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
from boxmot import BYTETracker, BoTSORT, DeepOCSORT, OCSORT, StrongSORT, HybridSORT


class TrackerNode(Node):
    def __init__(self):
        super().__init__("tracker_node")
        self.declare_parameter("yolo_model", "yolov8n.pt")
        self.declare_parameter("input_topic", "image_raw")
        self.declare_parameter("result_topic", "yolo_result")
        self.declare_parameter("result_image_topic", "yolo_image")
        self.declare_parameter("conf_thres", 0.25)
        self.declare_parameter("iou_thres", 0.45)
        self.declare_parameter("max_det", 300)
        self.declare_parameter("classes", list(range(80)))
        self.declare_parameter("tracker", "bytetrack.yaml")
        self.declare_parameter("device", "cpu")
        self.declare_parameter("result_conf", True)
        self.declare_parameter("result_line_width", 1)
        self.declare_parameter("result_font_size", 1)
        self.declare_parameter("result_font", "Arial.ttf")
        self.declare_parameter("result_labels", True)
        self.declare_parameter("result_boxes", True)
        # BoxMOT parameters
        self.declare_parameter("tracking_method", "ultralytics")  # ultralytics or boxmot
        self.declare_parameter("boxmot_tracker", "deepocsort")  # bytetrack, botsort, deepocsort, ocsort, strongsort, hybridsort
        self.declare_parameter("reid_model", "osnet_x0_25_msmt17.pt")  # ReID model for DeepOCSORT/StrongSORT
        self.declare_parameter("track_high_thresh", 0.5)
        self.declare_parameter("track_low_thresh", 0.1)
        self.declare_parameter("new_track_thresh", 0.6)
        self.declare_parameter("track_buffer", 30)
        self.declare_parameter("match_thresh", 0.8)

        path = get_package_share_directory("ultralytics_ros")
        yolo_model = self.get_parameter("yolo_model").get_parameter_value().string_value
        self.model = YOLO(f"{path}/models/{yolo_model}")
        self.model.fuse()

        self.bridge = cv_bridge.CvBridge()
        self.use_segmentation = yolo_model.endswith("-seg.pt")

        # Initialize BoxMOT tracker if needed
        tracking_method = self.get_parameter("tracking_method").get_parameter_value().string_value
        self.use_boxmot = tracking_method == "boxmot"
        self.boxmot_tracker = None

        if self.use_boxmot:
            self._initialize_boxmot_tracker()
            self.get_logger().info(f"Initialized BoxMOT tracker: {self.get_parameter('boxmot_tracker').get_parameter_value().string_value}")

        input_topic = (
            self.get_parameter("input_topic").get_parameter_value().string_value
        )
        result_topic = (
            self.get_parameter("result_topic").get_parameter_value().string_value
        )
        result_image_topic = (
            self.get_parameter("result_image_topic").get_parameter_value().string_value
        )
        self.create_subscription(Image, input_topic, self.image_callback, 1)
        self.results_pub = self.create_publisher(YoloResult, result_topic, 1)
        self.result_image_pub = self.create_publisher(Image, result_image_topic, 1)

    def _initialize_boxmot_tracker(self):
        """Initialize BoxMOT tracker based on selected algorithm"""
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

        if tracker_type == "bytetrack":
            self.boxmot_tracker = BYTETracker(**tracker_args)
        elif tracker_type == "botsort":
            self.boxmot_tracker = BoTSORT(**tracker_args)
        elif tracker_type == "deepocsort":
            reid_model = self.get_parameter("reid_model").get_parameter_value().string_value
            self.boxmot_tracker = DeepOCSORT(
                model_weights=reid_model,
                device=device,
                **tracker_args
            )
        elif tracker_type == "ocsort":
            self.boxmot_tracker = OCSORT(**tracker_args)
        elif tracker_type == "strongsort":
            reid_model = self.get_parameter("reid_model").get_parameter_value().string_value
            self.boxmot_tracker = StrongSORT(
                model_weights=reid_model,
                device=device,
                **tracker_args
            )
        elif tracker_type == "hybridsort":
            self.boxmot_tracker = HybridSORT(**tracker_args)
        else:
            self.get_logger().error(f"Unknown tracker type: {tracker_type}")
            self.boxmot_tracker = None

    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        conf_thres = self.get_parameter("conf_thres").get_parameter_value().double_value
        iou_thres = self.get_parameter("iou_thres").get_parameter_value().double_value
        max_det = self.get_parameter("max_det").get_parameter_value().integer_value
        classes = (
            self.get_parameter("classes").get_parameter_value().integer_array_value
        )
        device = self.get_parameter("device").get_parameter_value().string_value or None

        if self.use_boxmot and self.boxmot_tracker is not None:
            # Use BoxMOT tracking
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

            # Convert YOLO detections to BoxMOT format and apply tracking
            if results is not None and len(results) > 0 and results[0].boxes is not None:
                dets = results[0].boxes.data.cpu().numpy()  # x1, y1, x2, y2, conf, cls
                if len(dets) > 0:
                    # Apply BoxMOT tracking: dets format should be (x1, y1, x2, y2, conf, cls)
                    tracks = self.boxmot_tracker.update(dets, cv_image)

                    # Update results with track IDs
                    if tracks is not None and len(tracks) > 0:
                        # tracks format: (x1, y1, x2, y2, track_id, conf, cls, ...)
                        results[0].boxes.id = tracks[:, 4]  # track IDs
        else:
            # Use Ultralytics built-in tracking
            tracker = self.get_parameter("tracker").get_parameter_value().string_value
            results = self.model.track(
                source=cv_image,
                conf=conf_thres,
                iou=iou_thres,
                max_det=max_det,
                classes=classes,
                tracker=tracker,
                device=device,
                verbose=False,
                retina_masks=True,
            )

        if results is not None:
            yolo_result_msg = YoloResult()
            yolo_result_image_msg = Image()
            yolo_result_msg.header = msg.header
            yolo_result_image_msg.header = msg.header
            yolo_result_msg.detections = self.create_detections_array(results)
            yolo_result_image_msg = self.create_result_image(results)
            if self.use_segmentation:
                yolo_result_msg.masks = self.create_segmentation_masks(results)
            self.results_pub.publish(yolo_result_msg)
            self.result_image_pub.publish(yolo_result_image_msg)

    def create_detections_array(self, results):
        detections_msg = Detection2DArray()
        bounding_box = results[0].boxes.xywh
        classes = results[0].boxes.cls
        confidence_score = results[0].boxes.conf
        for bbox, cls, conf in zip(bounding_box, classes, confidence_score):
            detection = Detection2D()
            detection.bbox.center.position.x = float(bbox[0])
            detection.bbox.center.position.y = float(bbox[1])
            detection.bbox.size_x = float(bbox[2])
            detection.bbox.size_y = float(bbox[3])
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = results[0].names.get(int(cls))
            hypothesis.hypothesis.score = float(conf)
            detection.results.append(hypothesis)
            detections_msg.detections.append(detection)
        return detections_msg

    def create_result_image(self, results):
        result_conf = self.get_parameter("result_conf").get_parameter_value().bool_value
        result_line_width = (
            self.get_parameter("result_line_width").get_parameter_value().integer_value
        )
        result_font_size = (
            self.get_parameter("result_font_size").get_parameter_value().integer_value
        )
        result_font = (
            self.get_parameter("result_font").get_parameter_value().string_value
        )
        result_labels = (
            self.get_parameter("result_labels").get_parameter_value().bool_value
        )
        result_boxes = (
            self.get_parameter("result_boxes").get_parameter_value().bool_value
        )
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
        masks_msg = []
        for result in results:
            if hasattr(result, "masks") and result.masks is not None:
                for mask_tensor in result.masks:
                    mask_numpy = (
                        np.squeeze(mask_tensor.data.to("cpu").detach().numpy()).astype(
                            np.uint8
                        )
                        * 255
                    )
                    mask_image_msg = self.bridge.cv2_to_imgmsg(
                        mask_numpy, encoding="mono8"
                    )
                    masks_msg.append(mask_image_msg)
        return masks_msg


def main(args=None):
    rclpy.init(args=args)
    node = TrackerNode()
    rclpy.spin(node)


if __name__ == "__main__":
    main()
