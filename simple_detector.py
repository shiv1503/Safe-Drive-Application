"""
Sleep Detector — simple variant (no calibration step)
-------------------------------------------------------
Same idea as detector.py, but uses a fixed EAR threshold (0.32) instead
of calibrating to your eyes first. Simpler, but less accurate across
different people/lighting. Prefer detector.py unless you specifically
want this version.

Run:
    python simple_detector.py

Requires the same files as detector.py — see README.md.
"""

import os
import sys
import time
import threading

import cv2
import dlib
from scipy.spatial import distance as dist
from imutils import face_utils

try:
    from playsound import playsound
except ImportError:
    playsound = None


PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
ALARM_PATH = "alarm.mp3"
EAR_THRESHOLD = 0.32
FRAME_THRESHOLD = 10


def eye_aspect_ratio(eye):
    a = dist.euclidean(eye[1], eye[5])
    b = dist.euclidean(eye[2], eye[4])
    c = dist.euclidean(eye[0], eye[3])
    return (a + b) / (2.0 * c)


def sound_alarm(is_active):
    if playsound is None:
        print("[WARN] 'playsound' isn't installed — alarm will be silent. "
              "Run: pip install playsound==1.2.2")
        return
    if not os.path.isfile(ALARM_PATH):
        print(f"[WARN] '{ALARM_PATH}' not found — alarm will be silent.")
        return
    while is_active[0]:
        try:
            playsound(ALARM_PATH)
        except Exception as e:
            print(f"[WARN] Couldn't play alarm sound: {e}")
            return
        time.sleep(5)


def require_file(path, what, hint):
    if not os.path.isfile(path):
        print(f"[ERROR] Missing {what}: '{path}' was not found in this folder.")
        print(f"        {hint}")
        sys.exit(1)


def main():
    require_file(
        PREDICTOR_PATH,
        "facial landmark model",
        "Download shape_predictor_68_face_landmarks.dat and place it next "
        "to this script (see README.md for a download link).",
    )

    print("[INFO] Loading facial landmark predictor...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(PREDICTOR_PATH)

    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam (index 0).")
        sys.exit(1)

    frame_counter = 0
    alarm_on = [False]

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
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
                    if not alarm_on[0]:
                        alarm_on[0] = True
                        t = threading.Thread(target=sound_alarm, args=(alarm_on,))
                        t.daemon = True
                        t.start()
                    cv2.putText(frame, "DROWSY / SLEEPING!", (50, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            else:
                frame_counter = 0
                alarm_on[0] = False

            cv2.putText(frame, f"EAR: {ear:.2f}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Sleep Detector (simple)", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    alarm_on[0] = False
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
