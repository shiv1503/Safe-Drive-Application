"""
Sleep Detector for Drivers
---------------------------
Real-time drowsiness detection using webcam eye-tracking (Eye Aspect Ratio).

Run:
    python detector.py

Requires (see requirements.txt):
    opencv-python, dlib, numpy, scipy, imutils, playsound

Also requires two files placed in the SAME folder as this script
(not included in the repo — see README.md for download links):
    shape_predictor_68_face_landmarks.dat
    alarm.mp3
"""

import os
import sys
import time
import threading

import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
from imutils import face_utils

try:
    from playsound import playsound
except ImportError:
    playsound = None


PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
ALARM_PATH = "alarm.mp3"
CALIBRATION_SECONDS = 5
FRAME_THRESHOLD = 10
FALLBACK_EAR_THRESHOLD = 0.3


def eye_aspect_ratio(eye):
    """Standard EAR formula: mean of two vertical eye distances over the
    horizontal eye distance. Drops sharply when the eye closes."""
    a = dist.euclidean(eye[1], eye[5])
    b = dist.euclidean(eye[2], eye[4])
    c = dist.euclidean(eye[0], eye[3])
    return (a + b) / (2.0 * c)


def sound_alarm(is_active):
    """Loops the alarm sound every 5s while is_active[0] is True.
    Runs on a background thread so it doesn't block the video loop.
    Fails silently (with a printed warning) if playsound or the
    alarm file isn't available, instead of crashing the whole app."""
    if playsound is None:
        print("[WARN] 'playsound' isn't installed — alarm will be silent. "
              "Run: pip install playsound==1.2.2")
        return
    if not os.path.isfile(ALARM_PATH):
        print(f"[WARN] '{ALARM_PATH}' not found — alarm will be silent. "
              "Add an alarm.mp3 file next to this script.")
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
        print("[ERROR] Could not open webcam (index 0). Check that a camera "
              "is connected and not in use by another application.")
        sys.exit(1)

    frame_counter = 0
    alarm_on = [False]  # list so the background thread can see updates
    last_drowsy = None

    # ---- Step 1: Calibration ----
    print("[INFO] Calibration: please keep your eyes OPEN for "
          f"{CALIBRATION_SECONDS} seconds...")
    ear_values = []
    start_time = time.time()

    while time.time() - start_time < CALIBRATION_SECONDS:
        ret, frame = cap.read()
        if not ret or frame is None:
            # FIX: the original script didn't check this, so a dropped
            # frame during calibration crashed cv2.cvtColor() below.
            continue

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
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            cap.release()
            cv2.destroyAllWindows()
            return

    cv2.destroyWindow("Calibration")

    if ear_values:
        ear_threshold = float(np.mean(ear_values)) * 0.75
    else:
        print("[WARN] No face detected during calibration — using a "
              f"fallback threshold of {FALLBACK_EAR_THRESHOLD}.")
        ear_threshold = FALLBACK_EAR_THRESHOLD

    print(f"[INFO] EAR threshold set to: {ear_threshold:.2f}")

    # ---- Step 2: Detection loop ----
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("[WARN] Lost webcam frame — stopping.")
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

            if ear < ear_threshold:
                frame_counter += 1

                if frame_counter >= FRAME_THRESHOLD:
                    if not alarm_on[0]:
                        alarm_on[0] = True
                        t = threading.Thread(target=sound_alarm, args=(alarm_on,))
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
                alarm_on[0] = False
                last_drowsy = None
                cv2.putText(frame, "Awake", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

            cv2.putText(frame, f"EAR: {ear:.2f}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Smart Sleep Detector", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
            break

    alarm_on[0] = False
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
