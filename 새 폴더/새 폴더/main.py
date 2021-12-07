import cv2
import mediapipe as mp
import numpy as np
import keyboard
import joblib
import os
from coordinate import *

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
landMark = None

LIVE_TEST = False

model = joblib.load('DronePose.pkl')

if LIVE_TEST:
    # For webcam input:
    cap = cv2.VideoCapture(0)
    with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            # Draw the pose annotation on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

            image = cv2.flip(image, 1)

            # 이 부분 쓰레딩 해야할 수도 있어용 --> 속도 보고 1초에 한번? 정도로
            # landMark를 모델의 칼럼의 맞게 수정
            if landMark:
                landMark.updateList(results.pose_landmarks)
            else:
                landMark = LandMarkList(results.pose_landmarks)
                pose_obj = Pose(landMark, model)

            # predict 하고
            pose_obj.updatePose()
            poseText = pose_obj.getPose()
            # 출력하고
            cv2.putText(image, poseText, (100, 100),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 1, cv2.LINE_AA)

            # Flip the image horizontally for a selfie-view display.
            cv2.imshow('MediaPipe Pose', image)
            if cv2.waitKey(5) & 0xFF == 27:
                break
            if keyboard.is_pressed('q'):
                print("q is pressed.")
                break
        cap.release()
else:
    # For static images:
    IMAGE_FILES = os.listdir('ImageDB/poseTest/')
    BG_COLOR = (192, 192, 192)  # gray
    with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.5) as pose:
        for idx, file in enumerate(IMAGE_FILES):
            image = cv2.imread('ImageDB/poseTest/'+file)
            image_height, image_width, _ = image.shape
            # Convert the BGR image to RGB before processing.
            results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            if not results.pose_landmarks:
                print('No pose')
                continue

            # Draw pose landmarks on the image.
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            
            # 이 부분 쓰레딩 해야할 수도 있어용 --> 속도 보고 1초에 한번? 정도로
            # landMark를 모델의 칼럼의 맞게 수정
            
            landMark = LandMarkList(results.pose_landmarks)
            landMark.updateList(results.pose_landmarks)
            pose_obj = Pose(landMark, model)

            # predict 하고
            pose_obj.updatePose()
            poseText = pose_obj.getPose()
            
            # 출력하고
            cv2.putText(image, poseText, (100, 100),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 1, cv2.LINE_AA)
            
            cv2.imshow('poseTest', image)
            cv2.waitKey(0)
            
