import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import gc
from speck_cnn_hybrid import IntegratedSecureSpeck

def generate_global_hist():
    images_dir = r"s:\NIIT\Sem 6\R&D\RAND\RnD\BCSS_512\train_512"
    key = b"SecureEngine2026"
    engine = IntegratedSecureSpeck(key)
    
    global_hist_plain = np.zeros((256, 1), dtype=np.float64)
    global_hist_enc = np.zeros((256, 1), dtype=np.float64)
    
    if not os.path.exists(images_dir):
        print(f"Directory not found: {images_dir}")
        return
        
    images = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    images = sorted(images)[:6000] 
    
    if len(images) == 0:
        print("No images found.")
        return
        
    print(f"Processing {len(images)} images to generate average histogram...")
    
    for i, img_name in enumerate(images):
        path = os.path.join(images_dir, img_name)
        raw = cv2.imread(path)
        if raw is None: continue
        
        # Plain Hist
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY) if len(raw.shape) == 3 else raw
        hist_p = cv2.calcHist([gray], [0], None, [256], [0, 256])
        global_hist_plain += hist_p
        
        # Enc Hist
        enc_img, _, _ = engine.encrypt_adaptive(raw)
        gray_enc = cv2.cvtColor(enc_img, cv2.COLOR_BGR2GRAY) if len(enc_img.shape) == 3 else enc_img
        hist_e = cv2.calcHist([gray_enc], [0], None, [256], [0, 256])
        global_hist_enc += hist_e
        
        if (i+1) % 50 == 0:
            print(f"Processed {i+1} / {len(images)}", flush=True)
            
        del raw, gray, hist_p, enc_img, gray_enc, hist_e
        gc.collect()
            
    # Normalize or take average
    global_hist_plain /= len(images)
    global_hist_enc /= len(images)
    
    plt.figure(figsize=(14, 6))
    
    # Plaintext Histogram
    plt.subplot(1, 2, 1)
    plt.bar(range(256), global_hist_plain.flatten(), color='blue', alpha=0.7, width=1)
    plt.title("Average Plaintext Image Histogram (6000 Test Cases)")
    plt.xlabel("Pixel Intensity (0-255)")
    plt.ylabel("Average Frequency")
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    
    # Encrypted Histogram
    plt.subplot(1, 2, 2)
    plt.bar(range(256), global_hist_enc.flatten(), color='red', alpha=0.7, width=1)
    plt.title("Average Encrypted Image Histogram (6000 Test Cases)")
    plt.xlabel("Pixel Intensity (0-255)")
    plt.ylabel("Average Frequency")
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    
    plt.suptitle("CNN-Integrated SPECK Algorithm: Global Histogram Comparison", fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cumulative_comparison_histogram.png")
    plt.savefig(output_path, dpi=300)
    print(f"\nSuccessfully saved combined histogram layout at: {output_path}")

if __name__ == "__main__":
    generate_global_hist()
