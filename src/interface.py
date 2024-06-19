#!/usr/bin/env python3

import cv2
import numpy as np
import rospy
from yolov8_ROS.msg import Yolo_Poses
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import threading

class PoseInterface:
    def __init__(self):
        self.bridge = CvBridge()
        rospy.Subscriber("yolov8_pose_pub", Yolo_Poses, self.pose_callback)  # 포즈 정보를 구독
        rospy.Subscriber("/usb_cam/image_raw/compressed", CompressedImage, self.image_callback)  # 이미지를 구독

        self.frame = None  # 수신한 프레임을 저장

        # OpenCV 창 설정
        cv2.namedWindow("Pose Monitoring", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

        # 주기적으로 OpenCV 창을 업데이트하는 스레드 시작
        self.display_thread = threading.Thread(target=self.update_display)
        self.display_thread.daemon = True
        self.display_thread.start()

    def image_callback(self, data):
        try:
            # 이미지를 디코딩하여 self.frame에 저장
            self.frame = self.bridge.compressed_imgmsg_to_cv2(data, "bgr8")
            rospy.loginfo(f"Received image frame with shape: {self.frame.shape}")
        except Exception as e:
            self.frame = None
            rospy.logerr(f"Failed to convert image: {e}")

    def pose_callback(self, data):
        if self.frame is None:
            rospy.logwarn("No image frame available for processing")
            return

        if not data.yolo_poses:
            rospy.logwarn("No poses detected in the current frame")
            return

        warning_needed = False
        valid_keypoints = []

        for pose in data.yolo_poses:
            keypoints = np.array(pose.keypoints).reshape(-1, 3)

            if len(keypoints) == 0:
                rospy.logwarn("Keypoints array is empty or has invalid size.")
                continue

            indices = [0, 15, 16]

            for idx in indices:
                if idx < len(keypoints):
                    x, y, conf = keypoints[idx]
                    if conf > 0.5:
                        valid_keypoints.append((x, y, idx))
                        if idx == 0:
                            head_x, head_y = x, y
                        elif idx == 15:
                            left_ankle_x, left_ankle_y = x, y
                        elif idx == 16:
                            right_ankle_x, right_ankle_y = x, y
                else:
                    rospy.logwarn(f"Index {idx} is out of bounds for keypoints array.")

            if len(valid_keypoints) == 3:
                average_ankle_x = np.mean([left_ankle_x, right_ankle_x])
                error_margin = 20

                rospy.loginfo(f"Head X: {head_x}, Average Ankle X: {average_ankle_x}")

                if abs(head_x - average_ankle_x) < error_margin:
                    warning_needed = True

                cv2.circle(self.frame, (int(head_x), int(head_y)), 5, (255, 0, 0), -1)
                cv2.circle(self.frame, (int(left_ankle_x), int(left_ankle_y)), 5, (0, 0, 255), -1)
                cv2.circle(self.frame, (int(right_ankle_x), int(right_ankle_y)), 5, (0, 255, 0), -1)

        if warning_needed:
            self.display_warning()

    def update_display(self):
        while not rospy.is_shutdown():
            if self.frame is not None:
                cv2.imshow("Pose Monitoring", self.frame)
                cv2.waitKey(1)
            rospy.sleep(0.1)  # 100ms 간격으로 디스플레이 업데이트

    def display_warning(self):
        if self.frame is not None:
            cv2.putText(self.frame, "You need to be careful", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

def main():
    rospy.init_node("pose_interface")
    interface = PoseInterface()
    rospy.spin()

if __name__ == '__main__':
    main()
