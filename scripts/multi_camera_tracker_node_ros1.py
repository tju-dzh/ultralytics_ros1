#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Multi-Camera Tracker Node for ROS 1 (Noetic)

Handles 4 cameras with 360° coverage for panoramic object detection and tracking.
"""

import rospy
import cv_bridge
import numpy as np
import rospkg
from sensor_msgs.msg import Image
from ultralytics import YOLO
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
from ultralytics_ros.msg import YoloResult
from boxmot import DeepOCSORT, BoTSORT, StrongSORT
import message_filters
import threading


class GlobalTracker:
    """Global tracker for managing tracks across multiple cameras"""

    def __init__(self):
        self.global_tracks = {}
        self.next_global_id = 1
        self.lock = threading.Lock()
        self.id_mapping = {}

    def update_tracks(self, camera_id, tracks, class_names, timestamp):
        """Update global tracks with detections from a specific camera"""
        with self.lock:
            local_to_global = {}

            if tracks is None or len(tracks) == 0:
                return local_to_global

            for track in tracks:
                x1, y1, x2, y2, local_id, conf, cls = track[:7]
                local_id = int(local_id)
                cls_name = class_names.get(int(cls), str(int(cls)))

                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                w = x2 - x1
                h = y2 - y1
                angle_offset = camera_id * 90

                key = (camera_id, local_id)
                if key in self.id_mapping:
                    global_id = self.id_mapping[key]
                    if global_id in self.global_tracks:
                        self.global_tracks[global_id]['last_seen'] = timestamp
                        self.global_tracks[global_id]['camera_id'] = camera_id
                        self.global_tracks[global_id]['bbox'] = [cx, cy, w, h, angle_offset]
                        self.global_tracks[global_id]['conf'] = conf
                    local_to_global[local_id] = global_id
                else:
                    matched_global_id = self._match_adjacent_camera(
                        camera_id, cls_name, cx, cy, w, h, timestamp
                    )

                    if matched_global_id is not None:
                        global_id = matched_global_id
                        self.id_mapping[key] = global_id
                        self.global_tracks[global_id]['last_seen'] = timestamp
                        self.global_tracks[global_id]['camera_id'] = camera_id
                        self.global_tracks[global_id]['bbox'] = [cx, cy, w, h, angle_offset]
                    else:
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

            self._cleanup_old_tracks(timestamp, timeout=2.0)
            return local_to_global

    def _match_adjacent_camera(self, camera_id, cls_name, cx, cy, w, h, timestamp, threshold=0.3):
        """Match detection with tracks from adjacent cameras"""
        adjacent_cameras = [(camera_id - 1) % 4, (camera_id + 1) % 4]

        best_match = None
        best_score = threshold

        for global_id, track_info in self.global_tracks.items():
            if track_info['camera_id'] not in adjacent_cameras:
                continue
            if track_info['class'] != cls_name:
                continue
            if timestamp - track_info['last_seen'] > 0.5:
                continue

            track_cx, track_cy, track_w, track_h, _ = track_info['bbox']
            size_similarity = min(w, track_w) / max(w, track_w) * min(h, track_h) / max(h, track_h)
            vertical_diff = abs(cy - track_cy) / max(cy, track_cy) if max(cy, track_cy) > 0 else 0
            score = size_similarity * (1 - vertical_diff)

            if score > best_score:
                best_score = score
                best_match = global_id

        return best_match

    def _cleanup_old_tracks(self, current_time, timeout=2.0):
        """Remove tracks that haven't been seen for a while"""
        to_remove = []
        for global_id, track_info in self.global_tracks.items():
            if current_time - track_info['last_seen'] > timeout:
                to_remove.append(global_id)

        for global_id in to_remove:
            del self.global_tracks[global_id]
            keys_to_remove = [k for k, v in self.id_mapping.items() if v == global_id]
            for key in keys_to_remove:
                del self.id_mapping[key]


class MultiCameraTrackerNode:
    """Multi-camera tracker node for 360° panoramic tracking (ROS 1)"""

    def __init__(self):
        rospy.init_node("multi_camera_tracker_node", anonymous=False)

        # Get parameters
        yolo_model = rospy.get_param("~yolo_model", "yolov8n.pt")
        self.num_cameras = rospy.get_param("~num_cameras", 4)
        input_topics = rospy.get_param("~input_topics",
                                      ["/camera0/image_raw", "/camera1/image_raw",
                                       "/camera2/image_raw", "/camera3/image_raw"])
        result_topic_prefix = rospy.get_param("~result_topic_prefix", "/yolo_result")
        result_image_topic_prefix = rospy.get_param("~result_image_topic_prefix", "/yolo_image")

        self.conf_thres = rospy.get_param("~conf_thres", 0.25)
        self.iou_thres = rospy.get_param("~iou_thres", 0.45)
        self.max_det = rospy.get_param("~max_det", 300)
        self.classes = rospy.get_param("~classes", list(range(80)))
        self.device = rospy.get_param("~device", "cpu")

        # Visualization parameters
        self.result_conf = rospy.get_param("~result_conf", True)
        self.result_line_width = rospy.get_param("~result_line_width", 1)
        self.result_font_size = rospy.get_param("~result_font_size", 1)
        self.result_font = rospy.get_param("~result_font", "Arial.ttf")
        self.result_labels = rospy.get_param("~result_labels", True)
        self.result_boxes = rospy.get_param("~result_boxes", True)

        # BoxMOT parameters
        self.boxmot_tracker_type = rospy.get_param("~boxmot_tracker", "deepocsort")
        self.reid_model = rospy.get_param("~reid_model", "osnet_x0_25_msmt17.pt")
        self.track_high_thresh = rospy.get_param("~track_high_thresh", 0.5)
        self.track_low_thresh = rospy.get_param("~track_low_thresh", 0.1)
        self.new_track_thresh = rospy.get_param("~new_track_thresh", 0.6)
        self.track_buffer = rospy.get_param("~track_buffer", 30)
        self.match_thresh = rospy.get_param("~match_thresh", 0.8)

        # Synchronization parameters
        use_time_sync = rospy.get_param("~use_time_sync", True)
        sync_queue_size = rospy.get_param("~sync_queue_size", 10)
        sync_slop = rospy.get_param("~sync_slop", 0.1)

        # Get package path
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path("ultralytics_ros")

        # Load YOLO model
        model_path = f"{pkg_path}/models/{yolo_model}"
        rospy.loginfo(f"Loading YOLO model: {model_path}")
        self.model = YOLO(model_path)
        self.model.fuse()

        self.bridge = cv_bridge.CvBridge()
        self.use_segmentation = yolo_model.endswith("-seg.pt")

        # Initialize global tracker
        self.global_tracker = GlobalTracker()

        # Initialize BoxMOT trackers for each camera
        self.trackers = [self._create_boxmot_tracker() for _ in range(self.num_cameras)]

        # Setup publishers
        self.results_pubs = []
        self.result_image_pubs = []
        for i in range(self.num_cameras):
            self.results_pubs.append(
                rospy.Publisher(f"{result_topic_prefix}_camera{i}", YoloResult, queue_size=1)
            )
            self.result_image_pubs.append(
                rospy.Publisher(f"{result_image_topic_prefix}_camera{i}", Image, queue_size=1)
            )

        # Setup subscribers
        if use_time_sync and self.num_cameras > 1:
            rospy.loginfo("Using time-synchronized multi-camera input")
            self.image_subs = [message_filters.Subscriber(topic, Image) for topic in input_topics]

            self.sync = message_filters.ApproximateTimeSynchronizer(
                self.image_subs, sync_queue_size, sync_slop
            )
            self.sync.registerCallback(self.synchronized_image_callback)
        else:
            rospy.loginfo("Using independent processing for each camera")
            for i, topic in enumerate(input_topics):
                rospy.Subscriber(topic, Image,
                               lambda msg, cam_id=i: self.single_image_callback(msg, cam_id),
                               queue_size=1)

        rospy.loginfo(f"Multi-camera tracker initialized with {self.num_cameras} cameras")

    def _create_boxmot_tracker(self):
        """Create a BoxMOT tracker instance"""
        tracker_args = {
            "track_high_thresh": self.track_high_thresh,
            "track_low_thresh": self.track_low_thresh,
            "new_track_thresh": self.new_track_thresh,
            "track_buffer": self.track_buffer,
            "match_thresh": self.match_thresh,
            "frame_rate": 30,
        }

        if self.boxmot_tracker_type == "deepocsort":
            return DeepOCSORT(model_weights=self.reid_model, device=self.device, **tracker_args)
        elif self.boxmot_tracker_type == "botsort":
            return BoTSORT(**tracker_args)
        elif self.boxmot_tracker_type == "strongsort":
            return StrongSORT(model_weights=self.reid_model, device=self.device, **tracker_args)
        else:
            rospy.logwarn(f"Unknown tracker type: {self.boxmot_tracker_type}, using DeepOCSORT")
            return DeepOCSORT(model_weights=self.reid_model, device=self.device, **tracker_args)

    def synchronized_image_callback(self, *msgs):
        """Callback for synchronized multi-camera images"""
        timestamp = rospy.Time.now().to_sec()
        for camera_id, msg in enumerate(msgs):
            self.process_camera_image(msg, camera_id, timestamp)

    def single_image_callback(self, msg, camera_id):
        """Callback for single camera image"""
        timestamp = rospy.Time.now().to_sec()
        self.process_camera_image(msg, camera_id, timestamp)

    def process_camera_image(self, msg, camera_id, timestamp):
        """Process image from a specific camera"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except cv_bridge.CvBridgeError as e:
            rospy.logerr(f"CV Bridge Error: {e}")
            return

        device = self.device if self.device else None

        # Run YOLO detection
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

        # Apply BoxMOT tracking
        if results is not None and len(results) > 0 and results[0].boxes is not None:
            dets = results[0].boxes.data.cpu().numpy()
            if len(dets) > 0:
                tracks = self.trackers[camera_id].update(dets, cv_image)

                if tracks is not None and len(tracks) > 0:
                    class_names = results[0].names
                    local_to_global = self.global_tracker.update_tracks(
                        camera_id, tracks, class_names, timestamp
                    )

                    global_ids = []
                    for track in tracks:
                        local_id = int(track[4])
                        global_id = local_to_global.get(local_id, local_id)
                        global_ids.append(global_id)

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
        """Create Detection2DArray from YOLO results"""
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
        """Create annotated result image"""
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
        """Create segmentation masks"""
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
        node = MultiCameraTrackerNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Error in multi-camera tracker node: {e}")


if __name__ == "__main__":
    main()
