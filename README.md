# Sleep Detector for Drivers

Real-time driver drowsiness detection using webcam eye-tracking (Eye Aspect
Ratio / EAR) with dlib's 68-point facial landmark model. Sounds an alarm
when eyes stay closed for too long.

## Why it wasn't running

The original repo had two Python files and no README, requirements file,
or model/asset files. Specifically:

- `shape_predictor_68_face_landmarks.dat` (dlib's face-landmark model,
  ~99MB) was never included — required for both scripts to even start.
  It's also too large for GitHub to accept in a normal commit (100MB
  hard limit), so this fix adds `download_model.py` to fetch it on
  demand instead of committing it.
- `alarm.mp3` was never included — the alarm crashed the app once
  triggered.
- No `requirements.txt`, so a fresh clone hit `ModuleNotFoundError`
  immediately.
- `dlib` needs CMake + a C++ compiler to build — a plain `pip install dlib`
  fails on most machines without those installed first.
- The main script was named `dectector 2.py` (misspelled, with a space) —
  `python dectector 2.py` fails without quoting.
- The calibration loop didn't check whether a frame was actually read
  before processing it, so a dropped webcam frame during calibration
  crashed with `cv2.error: !ssize.empty()`.

All of the code-level issues are fixed in this version. The two binary
assets (model + alarm sound) can't be included in a text-based fix —
you need to download/add them yourself (links below).

## Setup

### 1. Install build tools for dlib (do this first)

- **Windows:** install [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
  with the "Desktop development with C++" workload, then `pip install cmake`.
- **macOS:** `brew install cmake`
- **Linux:** `sudo apt install cmake build-essential`

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Get the face-landmark model

This file is ~100MB — too big to commit to GitHub (it hard-blocks files
over 100MB), so it's `.gitignore`'d here and fetched on demand instead:

```bash
python download_model.py
```

This downloads and extracts `shape_predictor_68_face_landmarks.dat` into
the project folder automatically. Safe to re-run — it skips the download
if the file already exists. Do this once after cloning.

### 4. Add an alarm sound

Any short mp3/wav works. Add it to the project folder and name it
`alarm.mp3`, or edit `ALARM_PATH` at the top of `detector.py` to match
whatever filename you use. This one's small enough to commit normally if
you want it version-controlled.

### 5. Run it

```bash
python detector.py
```

You'll be asked to keep your eyes open for 5 seconds to calibrate, then
live detection starts. Press **ESC** to quit.

There's also `simple_detector.py` — the same idea without the calibration
step, using a fixed EAR threshold. Less accurate across different people
and lighting, kept here for reference.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `RuntimeError: Unable to open shape_predictor_68_face_landmarks.dat` | File missing or misnamed — see step 3 above |
| Script exits immediately with `[ERROR] Could not open webcam` | Camera in use by another app, or wrong index — try `cv2.VideoCapture(1)` if you have multiple cameras |
| `pip install dlib` fails with a compiler error | You skipped step 1 — install CMake + a C++ compiler first |
| No window appears / `cv2.imshow` errors about GUI support | You likely have `opencv-python-headless` installed instead of `opencv-python` — uninstall the headless version |
| Alarm never plays but everything else works | `alarm.mp3` missing, or `playsound` failed to install — check the console for a `[WARN]` message, the app now keeps running silently instead of crashing |

## How it works

1. **Calibration (5s):** measures your baseline EAR (eye openness) with
   eyes open, and sets the drowsy threshold at 75% of that baseline —
   this adapts to different people instead of using one fixed number.
2. **Detection loop:** each frame, detects your face and 68 landmarks,
   computes EAR for both eyes, and if it stays below threshold for
   10 consecutive frames, triggers a looping alarm on a background
   thread and shows a "DROWSY / SLEEPING!" overlay.
