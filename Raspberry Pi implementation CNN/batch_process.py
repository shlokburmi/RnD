"""
batch_process.py  — Raspberry Pi 4 (1 GB RAM) optimised build
Runs the CNN-SPECK hybrid vs standard SPECK on every image in ./images/
and writes a detailed result table to cnnresults.txt.
"""

import os
import gc
import cv2
import time
import numpy as np
import hashlib
import math
from datetime import datetime
from speck_cnn_hybrid import IntegratedSecureSpeck
from speck_vectorized import VectorizedSPECK

# ── Pi-specific global: cap OpenCV threads to avoid RAM pressure ──────────────
cv2.setNumThreads(2)

# ── CONFIGURE YOUR DATASET PATH HERE ────────────────────────────────────────
# Change this to the FULL path of your dataset folder on the Raspberry Pi.
# Examples:
#   IMAGES_DIR = "/home/pi/dataset"          ← absolute path (recommended)
#   IMAGES_DIR = "images"                    ← folder inside this project dir
#   IMAGES_DIR = "/media/pi/USB/my_dataset" ← USB drive
#
# You can also override at runtime without editing this file:
#   IMAGES_DIR=/home/pi/my_images python3 batch_process.py
# ─────────────────────────────────────────────────────────────────────────────
IMAGES_DIR   = os.environ.get("IMAGES_DIR", "images")   # ← CHANGE "images" to your path

RESULTS_FILE = "cnnresults.txt"
KEY          = b"SecureEngine2026"
IMAGE_EXTS   = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')


# ─────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────────────────────────────────────

def safe_resize(img, max_pixels=MAX_PIXELS):
    """
    Downscale image if it exceeds max_pixels while preserving aspect ratio.
    Works in-place on the returned array — no extra copy kept.
    """
    h, w = img.shape[:2]
    if w * h > max_pixels:
        scale = (max_pixels / (w * h)) ** 0.5
        img = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))),
                         interpolation=cv2.INTER_AREA)
    return img


def calculate_entropy(image_data):
    """Shannon entropy of pixel data (bits/pixel)."""
    if not isinstance(image_data, np.ndarray):
        image_data = np.frombuffer(image_data, dtype=np.uint8)
    flat = image_data.flatten()
    if len(flat) == 0:
        return 0.0
    counts = np.bincount(flat, minlength=256)
    probs  = counts[counts > 0] / len(flat)
    return float(-np.sum(probs * np.log2(probs)))


def calculate_avalanche(engine, img_path):
    """
    Bit-change ratio when one LSB of the centre pixel is flipped.
    Uses a temp file written to /tmp to avoid cluttering the project dir.
    """
    img = cv2.imread(img_path)
    if img is None:
        return 0.0
    img = safe_resize(img)

    c1, _, _ = engine.encrypt_adaptive(img_path)

    img_mod = img.copy()
    mid_r, mid_c = img_mod.shape[0] // 2, img_mod.shape[1] // 2
    if len(img_mod.shape) == 3:
        img_mod[mid_r, mid_c, 0] ^= 1
    else:
        img_mod[mid_r, mid_c] ^= 1

    temp_path = "/tmp/speck_temp_mod.png"
    cv2.imwrite(temp_path, img_mod)
    del img_mod  # free immediately

    c2, _, _ = engine.encrypt_adaptive(temp_path)
    try:
        os.remove(temp_path)
    except OSError:
        pass

    if c1 is None or c2 is None:
        return 0.0

    diff         = np.bitwise_xor(c1, c2)
    changed_bits = bin(int.from_bytes(diff.tobytes(), 'little')).count('1')
    total_bits   = c1.size * 8
    del c1, c2, diff
    return (changed_bits / total_bits) * 100.0


def run_standard_speck(image_path, key):
    """Full-image standard SPECK encryption, returns (duration_s, entropy)."""
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return 0.0, 0.0
    img = safe_resize(img)

    start  = time.perf_counter()
    cipher = VectorizedSPECK(hashlib.sha256(key).digest(), key_size=256)
    enc    = cipher.encrypt(img.tobytes())
    dur    = time.perf_counter() - start

    ent = calculate_entropy(enc)
    del img, enc, cipher
    return dur, ent


# ─────────────────────────────────────────────────────────────────────────────
# Main batch runner
# ─────────────────────────────────────────────────────────────────────────────

def run_comprehensive_batch():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir  = os.path.join(current_dir, IMAGES_DIR)
    results_file = os.path.join(current_dir, RESULTS_FILE)

    if not os.path.isdir(images_dir):
        print(f"[ERROR] Images directory not found: {images_dir}")
        return

    images = sorted(f for f in os.listdir(images_dir)
                    if f.lower().endswith(IMAGE_EXTS))

    if not images:
        print(f"[ERROR] No images found in {images_dir}")
        return

    hybrid_engine = IntegratedSecureSpeck(KEY)
    total = len(images)

    print(f"\n{'='*60}")
    print(f"  CNN-SPECK Batch Processor — Raspberry Pi 4 (1 GB)")
    print(f"  Images : {total}  |  Started: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"{'='*60}\n")

    with open(results_file, "w", buffering=1) as f:   # line-buffered: survives a crash mid-run
        # ── Header ─────────────────────────────────────────────────────────
        f.write("CNN-INTEGRATED VECTORIZED SPECK VS STANDARD SPECK — RASPBERRY PI 4 RESULTS\n")
        f.write(f"Run Date : {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        f.write(f"Images   : {total}\n")
        f.write(f"Key      : {KEY.decode()}\n")
        f.write("=" * 92 + "\n")
        f.write(f"{'Image':<20} | {'Type':<8} | {'Time(s)':<9} | {'Entropy':<8} | {'Avalanche':<12} | {'Pixels'}\n")
        f.write("-" * 92 + "\n")
        f.flush()

        for idx, image_name in enumerate(images, 1):
            img_path = os.path.join(images_dir, image_name)
            print(f"[{idx:>3}/{total}] Processing: {image_name}")

            try:
                # ── Read & resize once, reuse for both pipelines ───────────
                raw = cv2.imread(img_path)
                if raw is None:
                    raise ValueError("cv2.imread returned None — file unreadable")
                raw = safe_resize(raw)
                pixel_count = raw.shape[0] * raw.shape[1]

                # ── Save resized copy to a temp path so both methods use same size ──
                tmp_img_path = f"/tmp/speck_proc_{idx}.png"
                cv2.imwrite(tmp_img_path, raw)
                del raw
                gc.collect()

                # ── Hybrid (CNN + Selective SPECK) ─────────────────────────
                enc_img, _, dur_h = hybrid_engine.encrypt_adaptive(tmp_img_path)
                if enc_img is None:
                    raise ValueError("encrypt_adaptive returned None")
                ent_h = calculate_entropy(enc_img)
                del enc_img
                gc.collect()

                aval_h = calculate_avalanche(hybrid_engine, tmp_img_path)
                gc.collect()

                # ── Standard SPECK (full-image) ────────────────────────────
                dur_s, ent_s = run_standard_speck(tmp_img_path, KEY)
                gc.collect()

                try:
                    os.remove(tmp_img_path)
                except OSError:
                    pass

                # ── Write results (flush after each image so file is always valid) ──
                f.write(f"{image_name:<20} | Hybrid   | {dur_h:<9.4f} | {ent_h:<8.4f} | {aval_h:<12.2f} | {pixel_count}\n")
                f.write(f"{'':<20} | Standard | {dur_s:<9.4f} | {ent_s:<8.4f} | {'~50.00 (block)':<12} |\n")
                f.write("-" * 92 + "\n")
                f.flush()

                print(f"         Hybrid  → {dur_h:.4f}s | Entropy {ent_h:.4f} | Avalanche {aval_h:.2f}%")
                print(f"         Standard→ {dur_s:.4f}s | Entropy {ent_s:.4f}")

            except Exception as e:
                msg = str(e)[:60]
                f.write(f"{image_name:<20} | ERROR    | {msg}\n")
                f.write("-" * 92 + "\n")
                f.flush()
                print(f"         [ERROR] {image_name}: {e}")

            gc.collect()   # force-free between every image

        # ── Footer ─────────────────────────────────────────────────────────
        f.write(f"\nCompleted: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        f.write(f"Total images processed: {total}\n")

    print(f"\n{'='*60}")
    print(f"  Done! Results saved to: {results_file}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    run_comprehensive_batch()
