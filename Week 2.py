#importing all necessary modules 
import cv2 as cv
import numpy as np
import mediapipe as mp
import time
import math
from pycaw.pycaw import AudioUtilities

#function for calculating and displaying frame rate per second in the video
def frame_rate_per_second(pTime, img):
    cTime=time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    fps_string = "FPS : "+str(int(fps))
    cv.putText(img, fps_string, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
    return pTime

#function for tracking detecting and tracking hands
def hand_tracking():
    success = True
    mpHands = mp.solutions.hands
    hands = mpHands.Hands()
    cap = cv.VideoCapture(0)
    pTime = 0
    while success:
        success, img = cap.read()
        rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        results = hands.process(rgb_img)
        pTime=frame_rate_per_second(pTime, img)
        if results.multi_hand_landmarks:
            for handlms in results.multi_hand_landmarks:
                mpDraw.draw_landmarks(img, handlms, mpHands.HAND_CONNECTIONS)
        cv.imshow("Video", img)
        if cv.waitKey(1) & 0xFF==27:
            break

#function for pose estimation 
def pose_estimation():
    success = True
    mpPose = mp.solutions.pose
    pose = mpPose.Pose()
    cap = cv.VideoCapture(0)
    pTime = 0
    while success:
        success, img = cap.read()
        rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        results = pose.process(rgb_img)
        pTime=frame_rate_per_second(pTime, img)
        if results.pose_landmarks:
            mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        cv.imshow("Video", img)
        if cv.waitKey(1) & 0xFF==27:
            break

#funtcion for detecting and enclosing the faces detcted in the frame
def face_detection():
    cap = cv.VideoCapture(0)
    mpFaceDetection = mp.solutions.face_detection
    mpDraw = mp.solutions.drawing_utils
    faceDetection = mpFaceDetection.FaceDetection()
    pTime = 0
    success = True
    while success:
        success, img = cap.read()
        rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        pTime = frame_rate_per_second(pTime, img)
        results = faceDetection.process(rgb_img)
        if results.detections:
            for id, detection in enumerate(results.detections):
                mpDraw.draw_detection(img, detection)

        cv.imshow("Video", img)
        if cv.waitKey(1) & 0xFF==27:
            break

#function for combined display of hand tracking, pose estimation and face mesh technique in one frame
def techniques_combined():
    mpHands = mp.solutions.hands
    hands = mpHands.Hands()
    mpPose = mp.solutions.pose
    pose = mpPose.Pose()
    mpFace = mp.solutions.face_mesh
    face_mesh = mpFace.FaceMesh(max_num_faces=3) #to ensure that maximum number of faces detected in the frame are 3 
    draw_spec = mpDraw.DrawingSpec(thickness = 1, circle_radius = 2)
    cap = cv.VideoCapture(0)
    success = True
    pTime = 0
    while success:
        success, img =cap.read()
        rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB) 
        pTime = frame_rate_per_second(pTime, img)
        result_hand = hands.process(rgb_img)
        result_pose = pose.process(rgb_img)
        result_mesh = face_mesh.process(rgb_img)
        if result_hand.multi_hand_landmarks:
            for handlms in result_hand.multi_hand_landmarks:
                mpDraw.draw_landmarks(img, handlms, mpHands.HAND_CONNECTIONS)
        if result_pose.pose_landmarks:
            mpDraw.draw_landmarks(img, result_pose.pose_landmarks, mpPose.POSE_CONNECTIONS)
        if result_mesh.multi_face_landmarks:
            for id in result_mesh.multi_face_landmarks:
                mpDraw.draw_landmarks(img, id, mpFace.FACEMESH_TESSELATION, draw_spec, draw_spec)
        cv.imshow("Video", img)
        if cv.waitKey(1) & 0xFF==27:
            break

#function for enabling the feature of volumne control through hand gestures
def volume_control():
    device = AudioUtilities.GetSpeakers()
    volume = device.EndpointVolume

    cap = cv.VideoCapture(0)

    mpHands = mp.solutions.hands
    hands = mpHands.Hands(max_num_hands = 1)
    mpDraw = mp.solutions.drawing_utils
    pTime = 0
    success = True
    while success:
        success, img = cap.read()
        rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        pTime = frame_rate_per_second(pTime, img)
        results = hands.process(rgb_img)
        if results.multi_hand_landmarks:
            for handlms in results.multi_hand_landmarks:
                for id, lm in enumerate(handlms.landmark):
                    h, w, c = img.shape
                    if id == 4:
                        cx1, cy1 = int(lm.x*w), int(lm.y*h)
                    elif id == 8:
                        cx2, cy2 = int(lm.x*w), int(lm.y*h)
                cv.circle(img, (cx1, cy1), 5, (255,0,255), 5, -1)
                cv.circle(img, (cx2, cy2), 5, (255,0,255), 5, -1)
                cv.line(img, (cx1, cy1), (cx2, cy2), (255,0,255), 5)
                length = math.hypot(cx2-cx1, cy2-cy1)
                vol = np.interp(length, [7, 150], [-45, 0])
                print(vol)
                volume.SetMasterVolumeLevel(vol, None)
                mpDraw.draw_landmarks(img, handlms, mpHands.HAND_CONNECTIONS)

        cv.imshow("Video", img)
        if cv.waitKey(1) & 0xFF == 27:
            break


mpDraw = mp.solutions.drawing_utils
#calling function on the basis of user input
work = int(input("Enter the corresponding number in : \n 1. Face Detection \n 2. Hand Detection \n 3. Pose Estimation \n 4. Hand Detection, Pose Estimation and Face Mesh combined \n 5. Sound control through Gestures\nEnter :"))
if work == 1:
    face_detection()
elif work == 2:
    hand_tracking()
elif work == 3:
    pose_estimation()
elif work == 4:
    hand_tracking()
elif work == 5:
    volume_control()