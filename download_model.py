"""
Downloads and extracts shape_predictor_68_face_landmarks.dat so you never
need to commit a ~100MB binary to git (GitHub blocks files over 100MB
anyway). Run this once after cloning the repo:

    python download_model.py

Safe to re-run — it skips the download if the file already exists.
"""

import bz2
import os
import urllib.request

MODEL_URL = (
    "https://raw.githubusercontent.com/davisking/dlib-models/master/"
    "shape_predictor_68_face_landmarks.dat.bz2"
)
COMPRESSED_PATH = "shape_predictor_68_face_landmarks.dat.bz2"
OUTPUT_PATH = "shape_predictor_68_face_landmarks.dat"


def main():
    if os.path.isfile(OUTPUT_PATH):
        print(f"[INFO] '{OUTPUT_PATH}' already exists — nothing to do.")
        return

    print(f"[INFO] Downloading model from {MODEL_URL} ...")
    print("[INFO] This is ~64MB compressed, may take a minute.")
    urllib.request.urlretrieve(MODEL_URL, COMPRESSED_PATH)

    print("[INFO] Extracting...")
    with bz2.BZ2File(COMPRESSED_PATH, "rb") as f_in, open(OUTPUT_PATH, "wb") as f_out:
        f_out.write(f_in.read())

    os.remove(COMPRESSED_PATH)
    print(f"[INFO] Done — saved to '{OUTPUT_PATH}'.")


if __name__ == "__main__":
    main()
