import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point  # Import for centroid coordinates
from cv_bridge import CvBridge
#from tracker import Tracker
import random


class DetectionPublisher(Node):
    def __init__(self, camera_index=0):
        super().__init__('detection_publisher')
        self.detected_publisher_ = self.create_publisher(Image, 'detected_area', 10)
        self.original_publisher_ = self.create_publisher(Image, 'original_frame', 10)
        self.centroid_publisher_ = self.create_publisher(Point, 'object_centroid_diff', 10)
        self.all_mask_publisher_ = self.create_publisher(Image, 'all_detected_mask', 10)
        self.timer = self.create_timer(0.1, self.timer_callback)
        self.model = YOLO("best_tomato2024.pt")
        self.names = self.model.model.names
        self.cap = cv2.VideoCapture(camera_index)  # Use camera by index
        self.bridge = CvBridge()

    def timer_callback(self):
        ret, im0 = self.cap.read()
        if not ret:
            self.get_logger().info("End of video or failed to read frame.")
            rclpy.shutdown()
            return

        results = self.model.predict(im0)
        annotator = Annotator(im0, line_width=2)
        detected_area = np.zeros_like(im0)
        all_detected_mask = np.zeros_like(im0)

        # Frame centroid
        frame_height, frame_width = im0.shape[:2]
        frame_centroid_x, frame_centroid_y = frame_width // 2, frame_height // 2

        # Draw frame centroid on the original frame
        cv2.circle(im0, (frame_centroid_x, frame_centroid_y), 10, (200, 50, 150), -1)

        # Define the target classes you want to detect only one instance of each
        target_classes = {"GTomato", "RTomato", "HTomato", "stem"}
        detected_classes = set()

        if len(results[0].boxes) > 0:
            clss = results[0].boxes.cls.cpu().tolist()
            boxes = results[0].boxes.xyxy.cpu().numpy()

            # Sort the boxes based on area to prioritize larger objects if needed
            box_areas = [(x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in boxes]
            sorted_indices = sorted(range(len(boxes)), key=lambda i: box_areas[i], reverse=True)

            for idx in sorted_indices:
                x1, y1, x2, y2 = boxes[idx]
                cls = clss[idx]
                class_name = self.names[int(cls)]

                color = colors(int(cls), True)
                txt_color = annotator.get_txt_color(color)

                # Draw bounding box and label for all detected objects on all_detected_mask
                annotator.box_label((x1, y1, x2, y2), color=color, txt_color=txt_color)
                cv2.rectangle(all_detected_mask, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

                # Highlight the detected object region inside the all_detected_mask
                object_region = im0[int(y1):int(y2), int(x1):int(x2)]
                mask_region = all_detected_mask[int(y1):int(y2), int(x1):int(x2)]
                blended_region = cv2.addWeighted(mask_region, 0.3, object_region, 0.7, 0)
                all_detected_mask[int(y1):int(y2), int(x1):int(x2)] = blended_region

                # Only proceed with target classes for detected_area and centroid calculation
                if class_name in target_classes and class_name not in detected_classes:
                    detected_classes.add(class_name)

                    # Draw bounding box on detected_area
                    cv2.rectangle(detected_area, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

                    # Fill the detected area inside the bounding box with a semi-transparent color
                    detected_area[int(y1):int(y2), int(x1):int(x2)] = cv2.addWeighted(
                        detected_area[int(y1):int(y2), int(x1):int(x2)], 0.3,
                        object_region, 0.7, 0
                    )

                    if class_name == "RTomato":
                        # Calculate centroid of the bounding box
                        cx = int((x1 + x2) / 2)
                        cy = int((y1 + y2) / 2)

                        # Draw centroid of the object (red dot)
                        cv2.circle(im0, (cx, cy), 5, (255, 200, 100), -1)
                        self.get_logger().info(f"Centroid of {class_name}: ({cx}, {cy})")

                        # Calculate the difference between object and frame centroids
                        diff_x = cx - frame_centroid_x
                        diff_y = cy - frame_centroid_y

                        # Publish the centroid difference
                        self.publish_centroid_difference(diff_x, diff_y)

                    # Stop processing if all target classes are detected
                    if len(detected_classes) == len(target_classes):
                        break

        # Combine the detected area with the original frame to make the object visible
        highlighted_frame = cv2.addWeighted(im0, 0.7, detected_area, 0.3, 0)

        # Publish the detected area with bounding boxes
        self.publish_detected_area(detected_area)
        # Publish the original frame with the red dot
        self.publish_original_frame(highlighted_frame)
        # Publish the mask with all detected objects
        self.publish_all_detected_mask(all_detected_mask)

    def publish_detected_area(self, detected_area):
        ros_image = self.bridge.cv2_to_imgmsg(detected_area, encoding="bgr8")
        self.detected_publisher_.publish(ros_image)
        self.get_logger().info("Published detected area.")

    def publish_original_frame(self, original_frame):
        ros_image = self.bridge.cv2_to_imgmsg(original_frame, encoding="bgr8")
        self.original_publisher_.publish(ros_image)
        self.get_logger().info("Published original frame.")
        
    def publish_all_detected_mask(self, all_detected_mask):
        ros_image = self.bridge.cv2_to_imgmsg(all_detected_mask, encoding="bgr8")
        self.all_mask_publisher_.publish(ros_image)
        self.get_logger().info("Published all detected mask.")

    def publish_centroid_difference(self, diff_x, diff_y):
        point_msg = Point()
        point_msg.x = float(diff_x)
        point_msg.y = float(diff_y)
        point_msg.z = 0.0
        self.centroid_publisher_.publish(point_msg)
        self.get_logger().info(f"Published centroid difference: x={diff_x}, y={diff_y}")


def main(args=None):
    rclpy.init(args=args)
    camera_index = 0  # Use the default camera; update index if using a different camera
    detection_publisher = DetectionPublisher(camera_index)
    rclpy.spin(detection_publisher)
    detection_publisher.cap.release()
    cv2.destroyAllWindows()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

