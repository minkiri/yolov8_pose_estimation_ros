#!/usr/bin/env python3

from collections import defaultdict
import cv2
import numpy as np
import rospy
from sensor_msgs.msg import CompressedImage
from yolov8_ROS.msg import Yolo_Poses, Pose
from cv_bridge import CvBridge
import torch
import os
import sys
import time
from pathlib import Path
import threading

# yolov8 path 설정
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0] / "ultralytics"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  

from ultralytics import YOLO

class YoloV8_ROS():
    def __init__(self):
        rospy.Subscriber(source, CompressedImage, self.Callback)
        self.pub = rospy.Publisher("yolov8_pose_pub", Yolo_Poses, queue_size=1)  # 큐 사이즈를 1로 설정
        self.weights = rospy.get_param("~weights")  # 모델 경로 [(예: model.pt)]
        self.conf = rospy.get_param("~conf", 0.25)  # 신뢰도 임계값 [(예: 0.25)]
        imgsz_h = rospy.get_param("~imgsz_h", 384)  # 32의 배수로 이미지 높이 설정
        imgsz_w = rospy.get_param("~imgsz_w", 256)  # 32의 배수로 이미지 너비 설정
        self.imgsz = (imgsz_h, imgsz_w)
        self.device = torch.device(rospy.get_param("~device", 'cpu'))  # 기본적으로 CPU 사용
        
        # YOLOv8 모델 로드
        self.model = YOLO(self.weights)
        self.model.to(self.device)  # 모델을 원하는 장치로 이동
        
        # 모델을 FP16으로 설정 (GPU 사용 시 유용)
        if self.device.type == 'cuda':
            self.model.half()

        # 포즈 이력 저장
        self.pose_history = defaultdict(lambda: [])
        
        # 마지막 디스플레이 시간 기록
        self.last_display_time = time.time()
    
    def Callback(self, data):
        bridge = CvBridge()
        
        # 디버깅: ROS 메시지로부터 이미지 디코딩
        try:
            frame = bridge.compressed_imgmsg_to_cv2(data, "bgr8")
            if frame is None:
                rospy.logerr("Failed to decode image.")
                return
        except Exception as e:
            rospy.logerr(f"Exception during image decoding: {e}")
            return
        
        msg = Yolo_Poses()
        msg.header = data.header
        
        # 디버깅: 프레임 크기 출력
        rospy.loginfo(f"Received frame with shape: {frame.shape}")

        # YOLOv8 포즈 추정 실행
        results = self.model.predict(frame, task='pose', conf=self.conf, device=self.device, imgsz=self.imgsz)

        if results:
            keypoints = results[0].keypoints
            
            # 디버깅: 모델 결과 로그 추가
            if keypoints is not None and len(keypoints) > 0:
                rospy.loginfo(f"Number of keypoints detected: {len(keypoints)}")
            else:
                rospy.logwarn("No keypoints detected by the YOLO model.")

            # 결과를 프레임에 시각화
            annotated_frame = results[0].plot()

            # 트랙 ID 확인 및 처리
            if results[0].boxes is not None and results[0].boxes.id is not None:
                track_ids = results[0].boxes.id.int().tolist()
            else:
                track_ids = [None] * len(keypoints)  # 트랙 ID가 없으면 None으로 채움

            for keypoint, track_id in zip(keypoints.xy, track_ids):  # `xy` 속성 사용
                # Pose 메시지 생성
                pose_msg = Pose()
                pose_msg.track_id = track_id if track_id is not None else -1  # 트랙 ID가 없으면 -1로 설정
                pose_msg.keypoints = keypoint.flatten().tolist()  # ROS 메시지를 위한 1D 리스트로 변환

                # 디버깅: 변환된 키포인트 출력
                rospy.loginfo(f"Track ID: {track_id}, Keypoints: {pose_msg.keypoints}")

                # 포즈 이력 저장
                if track_id is not None:
                    self.pose_history[track_id].append(keypoint)
                    if len(self.pose_history[track_id]) > 90:
                        self.pose_history[track_id].pop(0)

                msg.yolo_poses.append(pose_msg)
                
        else:
            rospy.log.warn("No results from YOLO model.")

        # 메시지 발행
        rospy.loginfo(f"Publishing message with {len(msg.yolo_poses)} poses")
        self.pub.publish(msg)

def run():
    global source
    rospy.init_node("yolov8_pose_ROS")
    source = rospy.get_param("~source")  # 토픽 이름 [(예: /camera/image)]
    detect = YoloV8_ROS()

    # 스피너를 사용하여 콜백을 별도 스레드에서 실행
    spinner = threading.Thread(target=rospy.spin)
    spinner.start()

if __name__ == '__main__':
    run()
