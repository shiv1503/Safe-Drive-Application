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
    return (A + B) / (2.0 * C)

# Alarm function
def sound_alarm():
    global alarm_on
    while alarm_on:
        playsound("alarm.mp3")   # keep alarm.mp3 in same folder
        time.sleep(5)

# Load dlib face detector & predictor
print("[INFO] Loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Eye landmark indexes
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

cap = cv2.VideoCapture(0)

# Variables
frame_counter = 0
alarm_on = False
EAR_THRESHOLD = None  # Auto-calibrated
FRAME_THRESHOLD = 10
drowsy_time = 0
last_drowsy = None

print("[INFO] Calibration: Please keep your eyes OPEN for 5 seconds...")

# Step 1: Calibration
ear_values = []
start_time = time.time()
while time.time() - start_time < 5:  # 5 sec calibration
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        ear = (eye_aspect_ratio(leftEye) + eye_aspect_ratio(rightEye)) / 2.0
        ear_values.append(ear)
        cv2.putText(frame, "Calibrating... Keep eyes open!", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Calibration", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyWindow("Calibration")

# Set threshold based on calibration
if ear_values:
    EAR_THRESHOLD = np.mean(ear_values) * 0.75  # 75% of open-eye EAR
else:
    EAR_THRESHOLD = 0.3  # fallback

print(f"[INFO] EAR threshold set to: {EAR_THRESHOLD:.2f}")

# Step 2: Detection Loop
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
        ear = (eye_aspect_ratio(leftEye) + eye_aspect_ratio(rightEye)) / 2.0

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

                if last_drowsy is None:
                    last_drowsy = time.time()
                else:
                    drowsy_time = time.time() - last_drowsy
                    cv2.putText(frame, f"Drowsy for {drowsy_time:.1f}s", (50, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            frame_counter = 0
            alarm_on = False
            last_drowsy = None
            cv2.putText(frame, "Awake", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        cv2.putText(frame, f"EAR: {ear:.2f}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Smart Sleep Detector", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
