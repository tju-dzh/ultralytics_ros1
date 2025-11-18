#!/usr/bin/env python
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

import rospy
import cv_bridge
import numpy as np
import rospkg
from sensor_msgs.msg import Image
from ultralytics import YOLO
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
from ultralytics_ros.msg import YoloResult
from boxmot import BYTETracker, BoTSORT, DeepOCSORT, OCSORT, StrongSORT, HybridSORT


class TrackerNode:
    def __init__(self):
        rospy.init_node("tracker_node", anonymous=False)

        # Get parameters
        yolo_model = rospy.get_param("~yolo_model", "yolov8n.pt")
        input_topic = rospy.get_param("~input_topic", "image_raw")
        result_topic = rospy.get_param("~result_topic", "yolo_result")
        result_image_topic = rospy.get_param("~result_image_topic", "yolo_image")
        self.conf_thres = rospy.get_param("~conf_thres", 0.25)
        self.iou_thres = rospy.get_param("~iou_thres", 0.45)
        self.max_det = rospy.get_param("~max_det", 300)
        self.classes = rospy.get_param("~classes", list(range(80)))
        self.tracker_name = rospy.get_param("~tracker", "bytetrack.yaml")
        self.device = rospy.get_param("~device", "cpu")
        self.result_conf = rospy.get_param("~result_conf", True)
        self.result_line_width = rospy.get_param("~result_line_width", 1)
        self.result_font_size = rospy.get_param("~result_font_size", 1)
        self.result_font = rospy.get_param("~result_font", "Arial.ttf")
        self.result_labels = rospy.get_param("~result_labels", True)
        self.result_boxes = rospy.get_param("~result_boxes", True)

        # BoxMOT parameters
        tracking_method = rospy.get_param("~tracking_method", "ultralytics")
        self.boxmot_tracker_type = rospy.get_param("~boxmot_tracker", "deepocsort")
        self.reid_model = rospy.get_param("~reid_model", "osnet_x0_25_msmt17.pt")
        self.track_high_thresh = rospy.get_param("~track_high_thresh", 0.5)
        self.track_low_thresh = rospy.get_param("~track_low_thresh", 0.1)
        self.new_track_thresh = rospy.get_param("~new_track_thresh", 0.6)
        self.track_buffer = rospy.get_param("~track_buffer", 30)
        self.match_thresh = rospy.get_param("~match_thresh", 0.8)

        # Get package path using rospkg
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path("ultralytics_ros")

        # Load YOLO model
        model_path = f"{pkg_path}/models/{yolo_model}"
        rospy.loginfo(f"Loading YOLO model: {model_path}")
        self.model = YOLO(model_path)
        self.model.fuse()

        self.bridge = cv_bridge.CvBridge()
        self.use_segmentation = yolo_model.endswith("-seg.pt")

        # Initialize BoxMOT tracker if needed
        self.use_boxmot = tracking_method == "boxmot"
        self.boxmot_tracker = None

        if self.use_boxmot:
            self._initialize_boxmot_tracker()
            rospy.loginfo(f"Initialized BoxMOT tracker: {self.boxmot_tracker_type}")
        else:
            rospy.loginfo(f"Using Ultralytics tracker: {self.tracker_name}")

        # Publishers
        self.results_pub = rospy.Publisher(result_topic, YoloResult, queue_size=1)
        self.result_image_pub = rospy.Publisher(result_image_topic, Image, queue_size=1)

        # Subscriber
        self.image_sub = rospy.Subscriber(input_topic, Image, self.image_callback, queue_size=1)

        rospy.loginfo("Tracker node initialized successfully")

    def _initialize_boxmot_tracker(self):
        """Initialize BoxMOT tracker based on selected algorithm"""
        tracker_args = {
            "track_high_thresh": self.track_high_thresh,
            "track_low_thresh": self.track_low_thresh,
            "new_track_thresh": self.new_track_thresh,
            "track_buffer": self.track_buffer,
            "match_thresh": self.match_thresh,
            "frame_rate": 30,
        }

        if self.boxmot_tracker_type == "bytetrack":
            self.boxmot_tracker = BYTETracker(**tracker_args)
        elif self.boxmot_tracker_type == "botsort":
            self.boxmot_tracker = BoTSORT(**tracker_args)
        elif self.boxmot_tracker_type == "deepocsort":
            self.boxmot_tracker = DeepOCSORT(
                model_weights=self.reid_model,
                device=self.device,
                **tracker_args
            )
        elif self.boxmot_tracker_type == "ocsort":
            self.boxmot_tracker = OCSORT(**tracker_args)
        elif self.boxmot_tracker_type == "strongsort":
            self.boxmot_tracker = StrongSORT(
                model_weights=self.reid_model,
                device=self.device,
                **tracker_args
            )
        elif self.boxmot_tracker_type == "hybridsort":
            self.boxmot_tracker = HybridSORT(**tracker_args)
        else:
            rospy.logerr(f"Unknown tracker type: {self.boxmot_tracker_type}")
            self.boxmot_tracker = None

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except cv_bridge.CvBridgeError as e:
            rospy.logerr(f"CV Bridge Error: {e}")
            return

        device = self.device if self.device else None

        if self.use_boxmot and self.boxmot_tracker is not None:
            # Use BoxMOT tracking
            results = self.model(
                source=cv_image,
                conf=self.conf_thres,
                iou=self.iou_thres,
                max_det=self.max_det,
                classes=self.classes,
                device=device,
                verbose=False,
                retina_masks=True,
            )

            # Convert YOLO detections to BoxMOT format and apply tracking
            if results is not None and len(results) > 0 and results[0].boxes is not None:
                dets = results[0].boxes.data.cpu().numpy()
                if len(dets) > 0:
                    # Apply BoxMOT tracking
                    tracks = self.boxmot_tracker.update(dets, cv_image)

                    # Update results with track IDs
                    if tracks is not None and len(tracks) > 0:
                        results[0].boxes.id = tracks[:, 4]
        else:
            # Use Ultralytics built-in tracking
            results = self.model.track(
                source=cv_image,
                conf=self.conf_thres,
                iou=self.iou_thres,
                max_det=self.max_det,
                classes=self.classes,
                tracker=self.tracker_name,
                device=device,
                verbose=False,
                retina_masks=True,
            )

        if results is not None:
            # Create and publish YoloResult message
            yolo_result_msg = YoloResult()
            yolo_result_msg.header = msg.header
            yolo_result_msg.detections = self.create_detections_array(results)

            # Create and publish result image
            yolo_result_image_msg = self.create_result_image(results)
            yolo_result_image_msg.header = msg.header

            # Add segmentation masks if available
            if self.use_segmentation:
                yolo_result_msg.masks = self.create_segmentation_masks(results)

            self.results_pub.publish(yolo_result_msg)
            self.result_image_pub.publish(yolo_result_image_msg)

    def create_detections_array(self, results):
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
            detections_msg.detections.append(detection)

        return detections_msg

    def create_result_image(self, results):
        plotted_image = results[0].plot(
            conf=self.result_conf,
            line_width=self.result_line_width,
            font_size=self.result_font_size,
            font=self.result_font,
            labels=self.result_labels,
            boxes=self.result_boxes,
        )
        result_image_msg = self.bridge.cv2_to_imgmsg(plotted_image, encoding="bgr8")
        return result_image_msg

    def create_segmentation_masks(self, results):
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

    def run(self):
        rospy.spin()


def main():
    try:
        node = TrackerNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Error in tracker node: {e}")


if __name__ == "__main__":
    main()
