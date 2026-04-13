"""
batch_process.py  — Raspberry Pi 4 (1 GB RAM) Comprehensive Research Benchmarking
Optimized for memory efficiency while calculating multiple security parameters.
"""

import os
import gc
import cv2
import time
import numpy as np
import hashlib
from datetime import datetime
from speck_cnn_hybrid import IntegratedSecureSpeck

# ── Pi-specific global: cap OpenCV threads to avoid RAM pressure ──────────────
cv2.setNumThreads(2)

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
MAX_PIXELS   = 1_000_000   # ~1 MP cap for RPi 1GB RAM stability
IMAGES_DIR   = os.environ.get("IMAGES_DIR", "/home/yyyuvvvraj/Desktop/R&D/RnD/BCSS_512/train_512")
RESULTS_FILE = "cnnresults.txt"
SUMMARY_FILE = "cumulative_report.txt"
KEY          = b"SecureEngine2026"
IMAGE_EXTS   = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

HIST_PLAIN_DIR = "histograms_plain"
HIST_ENC_DIR   = "histograms_encrypted"

# ─────────────────────────────────────────────────────────────────────────────
# Helper utilities for Security Metrics
# ─────────────────────────────────────────────────────────────────────────────

def safe_resize(img, max_pixels=MAX_PIXELS):
    h, w = img.shape[:2]
    if w * h > max_pixels:
        scale = (max_pixels / (w * h)) ** 0.5
        img = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))),
                         interpolation=cv2.INTER_AREA)
    return img

def calculate_entropy(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    flat = image.flatten()
    if len(flat) == 0: return 0.0
    counts = np.bincount(flat, minlength=256)
    probs = counts[counts > 0] / len(flat)
    return float(-np.sum(probs * np.log2(probs)))

def calculate_psnr(original, encrypted):
    mse = np.mean((original.astype(np.float32) - encrypted.astype(np.float32)) ** 2)
    if mse == 0: return 100.0
    return 20 * np.log10(255.0 / np.sqrt(mse))

def calculate_npcr(original, encrypted):
    diff = (original != encrypted).astype(np.float32)
    return (np.sum(diff) / original.size) * 100.0

def calculate_uaci(original, encrypted):
    diff = np.abs(original.astype(np.float32) - encrypted.astype(np.float32))
    return (np.sum(diff) / (original.size * 255.0)) * 100.0

def calculate_correlation(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    img_data = image.astype(np.float32)
    h, w = img_data.shape
    
    # Select 3000 random pixel pairs for correlation
    n = 3000
    if h*w < n+1: n = (h*w // 2) - 1
    
    # Horizontal
    x_h = img_data[:, :-1].flatten()
    y_h = img_data[:, 1:].flatten()
    idx_h = np.random.choice(len(x_h), n, replace=False)
    corr_h = np.corrcoef(x_h[idx_h], y_h[idx_h])[0, 1]
    
    # Vertical
    x_v = img_data[:-1, :].flatten()
    y_v = img_data[1:, :].flatten()
    idx_v = np.random.choice(len(x_v), n, replace=False)
    corr_v = np.corrcoef(x_v[idx_v], y_v[idx_v])[0, 1]
    
    # Diagonal
    x_d = img_data[:-1, :-1].flatten()
    y_d = img_data[1:, 1:].flatten()
    idx_d = np.random.choice(len(x_d), n, replace=False)
    corr_d = np.corrcoef(x_d[idx_d], y_d[idx_d])[0, 1]
    
    return corr_h, corr_v, corr_d

def save_histogram_image(image, output_path):
    # Manual drawing of histogram using OpenCV to avoid dependency on matplotlib
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist_h, hist_w = 400, 512
    # Create a canvas
    canvas = np.ones((hist_h, hist_w, 3), dtype=np.uint8) * 255
    
    # Normalize histogram to fit in the canvas height
    cv2.normalize(hist, hist, 0, hist_h - 20, cv2.NORM_MINMAX)
    
    bin_w = hist_w // 256
    for i in range(1, 256):
        cv2.line(canvas, 
                 (bin_w * (i - 1), hist_h - int(float(hist[i-1][0]))),
                 (bin_w * i, hist_h - int(float(hist[i][0]))),
                 (0, 0, 255), 2)
    
    cv2.imwrite(output_path, canvas)

# ─────────────────────────────────────────────────────────────────────────────
# Main batch runner
# ─────────────────────────────────────────────────────────────────────────────

def run_comprehensive_batch():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir  = os.path.join(current_dir, IMAGES_DIR)
    results_file = os.path.join(current_dir, RESULTS_FILE)
    
    os.makedirs(HIST_PLAIN_DIR, exist_ok=True)
    os.makedirs(HIST_ENC_DIR, exist_ok=True)

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
    
    summary_data = {
        'enc_times': [], 'dec_times': [], 'entropy': [],
        'psnr': [], 'npcr': [], 'uaci': [],
        'corr_h': [], 'corr_v': [], 'corr_d': []
    }

    print(f"\nStarting Benchmarking of {total} images...")

    with open(results_file, "w", buffering=1) as f:
        f.write("CNN-INTEGRATED SPECK COMPREHENSIVE SECURITY REPORT\n")
        f.write("-" * 120 + "\n")
        f.write(f"{'Image':<15} | {'Enc_T(s)':<8} | {'Dec_T(s)':<8} | {'Entp':<6} | {'NPCR%':<6} | {'UACI%':<6} | {'PSNR':<6} | {'Corr_H':<6} | {'Corr_V':<6} | {'Corr_D':<6}\n")
        f.write("-" * 120 + "\n")

        for idx, image_name in enumerate(images, 1):
            img_path = os.path.join(images_dir, image_name)
            print(f"[{idx}/{total}] {image_name}")

            try:
                # 1. Plain Image
                raw = cv2.imread(img_path)
                if raw is None: continue
                raw = safe_resize(raw)
                
                # Save plain histogram
                save_histogram_image(raw, os.path.join(HIST_PLAIN_DIR, f"{image_name}_hist.png"))

                # 2. Encryption
                enc_img, mask, enc_time = hybrid_engine.encrypt_adaptive(raw)
                
                # Save encrypted histogram
                save_histogram_image(enc_img, os.path.join(HIST_ENC_DIR, f"{image_name}_hist_enc.png"))

                # 3. Decryption
                dec_img, dec_time = hybrid_engine.decrypt_adaptive(enc_img, mask)

                # 4. Metrics
                entropy = calculate_entropy(enc_img)
                psnr = calculate_psnr(raw, enc_img)
                npcr = calculate_npcr(raw, enc_img)
                uaci = calculate_uaci(raw, enc_img)
                ch, cv, cd = calculate_correlation(enc_img)

                # Store for summary
                summary_data['enc_times'].append(enc_time)
                summary_data['dec_times'].append(dec_time)
                summary_data['entropy'].append(entropy)
                summary_data['psnr'].append(psnr)
                summary_data['npcr'].append(npcr)
                summary_data['uaci'].append(uaci)
                summary_data['corr_h'].append(ch)
                summary_data['corr_v'].append(cv)
                summary_data['corr_d'].append(cd)

                # Write detail
                f.write(f"{image_name:<15} | {enc_time:<8.4f} | {dec_time:<8.4f} | {entropy:<6.4f} | {npcr:<6.2f} | {uaci:<6.2f} | {psnr:<6.2f} | {ch:<6.3f} | {cv:<6.3f} | {cd:<6.3f}\n")
                
                del raw, enc_img, dec_img, mask
                gc.collect()

            except Exception as e:
                print(f"Error on {image_name}: {e}")

    # Generate Cumulative Report
    with open(SUMMARY_FILE, "w") as sf:
        sf.write("CUMULATIVE RESEARCH COMPARISON REPORT\n")
        sf.write("="*40 + "\n")
        sf.write(f"Total Images Processed: {len(summary_data['enc_times'])}\n\n")
        
        sf.write("AVERAGE PERFORMANCE METRICS:\n")
        sf.write(f"- Avg Encryption Time: {np.mean(summary_data['enc_times']):.4f} s\n")
        sf.write(f"- Avg Decryption Time: {np.mean(summary_data['dec_times']):.4f} s\n")
        sf.write(f"- Avg Entropy:        {np.mean(summary_data['entropy']):.4f}\n")
        sf.write(f"- Avg NPCR:           {np.mean(summary_data['npcr']):.2f} %\n")
        sf.write(f"- Avg UACI:           {np.mean(summary_data['uaci']):.2f} %\n")
        sf.write(f"- Avg PSNR:           {np.mean(summary_data['psnr']):.2f} dB\n\n")
        
        sf.write("CORRELATION ANALYSIS (Average):\n")
        sf.write(f"- Horizontal: {np.mean(summary_data['corr_h']):.4f}\n")
        sf.write(f"- Vertical:   {np.mean(summary_data['corr_v']):.4f}\n")
        sf.write(f"- Diagonal:   {np.mean(summary_data['corr_d']):.4f}\n")

    print(f"\nCompleted! Results: {RESULTS_FILE}, Summary: {SUMMARY_FILE}")
    print(f"Histograms: {HIST_PLAIN_DIR}/ and {HIST_ENC_DIR}/")

if __name__ == "__main__":
    run_comprehensive_batch()
