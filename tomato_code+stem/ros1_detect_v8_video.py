import rospy
import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
import os


class DetectionPublisher:
    def __init__(self, video_path):
        rospy.init_node('detection_publisher', anonymous=True)

        self.detected_publisher_ = rospy.Publisher('detected_area', Image, queue_size=10)
        self.original_publisher_ = rospy.Publisher('original_frame', Image, queue_size=10)
        self.centroid_publisher_ = rospy.Publisher('object_centroid_diff', Point, queue_size=10)
        self.stem_publisher_ = rospy.Publisher('stem_centroid', Point, queue_size=10)
        self.all_mask_publisher_ = rospy.Publisher('all_detected_mask', Image, queue_size=10)

        self.model = YOLO("best_final.pt")
        self.names = self.model.model.names
        self.cap = cv2.VideoCapture(video_path)
        self.bridge = CvBridge()

        # Variables for blinking detection
        self.previous_bounding_box = None
        self.previous_bounding_box_stem = None
        self.blinking_threshold = 20
        self.blinking_timespan = 5 * 10

        # Loop rate
        self.rate = rospy.Rate(10)  # 10 Hz

    def run(self):
        while not rospy.is_shutdown():
            ret, im0 = self.cap.read()
            if not ret:
                rospy.loginfo("End of video or failed to read frame.")
                break

            results = self.model.predict(im0)
            annotator = Annotator(im0, line_width=2)
            detected_area = np.zeros_like(im0)
            all_detected_mask = np.zeros_like(im0)

            frame_height, frame_width = im0.shape[:2]
            frame_centroid_x, frame_centroid_y = frame_width // 2, frame_height // 2

            cv2.circle(im0, (frame_centroid_x, frame_centroid_y), 10, (200, 50, 150), -1)

            target_classes = {"0", "1", "2", "3"}
            detected_classes = set()

            if len(results[0].boxes) > 0:
                clss = results[0].boxes.cls.cpu().tolist()
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confs = results[0].boxes.conf.cpu().numpy()

                conf_threshold = 0.00000007
                filtered_indices = [i for i, conf in enumerate(confs) if conf >= conf_threshold]

                filtered_boxes = [(boxes[i], clss[i], confs[i]) for i in filtered_indices]
                filtered_boxes.sort(key=lambda b: (b[0][2] - b[0][0]) * (b[0][3] - b[0][1]), reverse=True)

                for box, cls, conf in filtered_boxes:
                    x1, y1, x2, y2 = box
                    class_name = self.names[int(cls)]

                    color = colors(int(cls), True)
                    txt_color = annotator.get_txt_color(color)

                    annotator.box_label((x1, y1, x2, y2), color=color, txt_color=txt_color)
                    cv2.rectangle(all_detected_mask, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

                    object_region = im0[int(y1):int(y2), int(x1):int(x2)]
                    mask_region = all_detected_mask[int(y1):int(y2), int(x1):int(x2)]
                    blended_region = cv2.addWeighted(mask_region, 0.3, object_region, 0.7, 0)
                    all_detected_mask[int(y1):int(y2), int(x1):int(x2)] = blended_region

                    if class_name in target_classes and class_name not in detected_classes:
                        detected_classes.add(class_name)
                        cv2.rectangle(detected_area, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        detected_area[int(y1):int(y2), int(x1):int(x2)] = cv2.addWeighted(
                            detected_area[int(y1):int(y2), int(x1):int(x2)], 0.3,
                            object_region, 0.7, 0
                        )

                        if class_name == "2":
                            if self.previous_bounding_box:
                                px1, py1, px2, py2 = self.previous_bounding_box
                                if (abs(x1 - px1) > self.blinking_threshold or
                                        abs(y1 - py1) > self.blinking_threshold or
                                        abs(x2 - px2) > self.blinking_threshold or
                                        abs(y2 - py2) > self.blinking_threshold):
                                    rospy.logwarn("BLINKING TOMATO")
                            self.previous_bounding_box = (x1, y1, x2, y2)
                            cx = int((x1 + x2) / 2)
                            cy = int((y1 + y2) / 2)
                            cv2.circle(im0, (cx, cy), 5, (255, 200, 100), -1)
                            rospy.loginfo(f"Centroid of {class_name}: ({cx}, {cy})")

                            diff_x = cx - frame_centroid_x
                            diff_y = cy - frame_centroid_y
                            self.publish_centroid_difference(diff_x, diff_y)

                        if class_name == "3":
                            cx_stem = int((x1 + x2) / 2)
                            cy_stem = int((y1 + y2) / 2)
                            cv2.circle(im0, (cx_stem, cy_stem), 5, (0, 255, 0), -1)
                            rospy.loginfo(f"Centroid of {class_name}: ({cx_stem}, {cy_stem})")
                            self.publish_stem_centroid(cx_stem, cy_stem)

                        if len(detected_classes) == len(target_classes):
                            break

            highlighted_frame = cv2.addWeighted(im0, 0.7, detected_area, 0.3, 0)

            self.publish_detected_area(detected_area)
            self.publish_original_frame(highlighted_frame)
            self.publish_all_detected_mask(all_detected_mask)

            self.rate.sleep()

        self.cap.release()
        cv2.destroyAllWindows()

    def publish_detected_area(self, detected_area):
        ros_image = self.bridge.cv2_to_imgmsg(detected_area, encoding="bgr8")
        self.detected_publisher_.publish(ros_image)

    def publish_original_frame(self, original_frame):
        ros_image = self.bridge.cv2_to_imgmsg(original_frame, encoding="bgr8")
        self.original_publisher_.publish(ros_image)

    def publish_all_detected_mask(self, all_detected_mask):
        ros_image = self.bridge.cv2_to_imgmsg(all_detected_mask, encoding="bgr8")
        self.all_mask_publisher_.publish(ros_image)

    def publish_centroid_difference(self, diff_x, diff_y):
        point_msg = Point()
        point_msg.x = float(diff_x)
        point_msg.y = float(diff_y)
        point_msg.z = 0.0
        self.centroid_publisher_.publish(point_msg)

    def publish_stem_centroid(self, cx_stem, cy_stem):
        point_msg = Point()
        point_msg.x = float(cx_stem)
        point_msg.y = float(cy_stem)
        point_msg.z = 0.0
        self.stem_publisher_.publish(point_msg)


def main():
    video_path = os.path.expanduser('~/Desktop/tomato/testAll.mp4')
    detection_publisher = DetectionPublisher(video_path)
    detection_publisher.run()


if __name__ == '__main__':
    main()

