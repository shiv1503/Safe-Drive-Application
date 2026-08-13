import cv2
import dlib
import time
import numpy as np
from scipy.spatial import distance as dist
from imutils import face_utils
from playsound import playsound
import threading

# EAR calculation
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Thresholds
EAR_THRESHOLD = 0.32
FRAME_THRESHOLD = 10

# Alarm function (loops every 5 seconds while alarm_on = True)
def sound_alarm():
    global alarm_on
    while alarm_on:
        playsound("alarm.mp3")   # keep an alarm sound (mp3/wav) in same folder
        time.sleep(5)

# Load dlib face detector & predictor
print("[INFO] Loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Get eye landmarks
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

cap = cv2.VideoCapture(0)
frame_counter = 0
alarm_on = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        leftHull = cv2.convexHull(leftEye)
        rightHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightHull], -1, (0, 255, 0), 1)

        if ear < EAR_THRESHOLD:
            frame_counter += 1

            if frame_counter >= FRAME_THRESHOLD:
                if not alarm_on:
                    alarm_on = True
                    t = threading.Thread(target=sound_alarm)
                    t.daemon = True
                    t.start()

                cv2.putText(frame, "DROWSY / SLEEPING!", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        else:
            frame_counter = 0
            alarm_on = False  # this stops the loop in sound_alarm

        cv2.putText(frame, f"EAR: {ear:.2f}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Sleep Detector with Looping Alarm", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
