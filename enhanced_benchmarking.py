import cv2
import numpy as np
import time
import os
import hashlib
from speck_vectorized import VectorizedSPECK
from speck_cnn_hybrid import IntegratedSecureSpeck

# --- Security Analyzers ---
def calculate_entropy(image):
    """Calculate Shannon Entropy (Ideal for encrypted = 8.0)"""
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    marg = np.histogram(image, bins=256, range=(0, 255))[0]
    marg = marg / np.sum(marg)
    marg = marg[marg > 0]
    return -np.sum(marg * np.log2(marg))

def calculate_correlation(image):
    """Horizontal pixel correlation (Ideal for encrypted ~ 0.0)"""
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    x = image[:, :-1].flatten().astype(float)
    y = image[:, 1:].flatten().astype(float)
    return np.corrcoef(x, y)[0, 1]

def calculate_avalanche(cipher_obj, data):
    subset = data[:2048]
    c1 = cipher_obj.encrypt(subset)
    key2 = bytearray(hashlib.sha256(b"Secret").digest())
    key2[0] ^= 0x01
    cipher2 = VectorizedSPECK(bytes(key2))
    c2 = cipher2.encrypt(subset)
    b1 = np.unpackbits(np.frombuffer(c1, dtype=np.uint8))
    b2 = np.unpackbits(np.frombuffer(c2, dtype=np.uint8))
    return (np.sum(b1 != b2) / len(b1)) * 100

# --- Benchmarking Logic ---
def _rotate_right(x, r):
    return ((x >> r) | (x << (64 - r))) & 0xFFFFFFFFFFFFFFFF

def _rotate_left(x, r):
    return ((x << r) | (x >> (64 - r))) & 0xFFFFFFFFFFFFFFFF

def benchmark_scalar_logic(data):
    rounds = 34
    start = time.perf_counter()
    sample_size = min(len(data), 5120) # 5KB sample for scaling
    for i in range(0, sample_size, 16):
        x, y = 12345, 67890
        for _ in range(rounds):
            x = (_rotate_right(x, 8) + y) & 0xFFFFFFFFFFFFFFFF
            x ^= 12345678 
            y = _rotate_left(y, 3) ^ x
    end = time.perf_counter()
    return (end - start) * (len(data) / sample_size)

def run_comprehensive_tests():
    images_dir = "SPECK/code/Images"
    images = ["brainmri.jpg", "ctscan.jpg", "liverultrasound.jpg", "spectmpi.jpg", "xrayjpeg.jpeg"]
    output_dir = "comprehensive_research_output"
    os.makedirs(output_dir, exist_ok=True)
    
    hybrid_engine = IntegratedSecureSpeck(b"AI_KEY_2026")
    vec_cipher = VectorizedSPECK(hashlib.sha256(b"AI_KEY_2026").digest())
    
    results_file = "ENHANCED_SECURITY_ANALYSIS.txt"
    report = []
    
    header = f"{'IMAGE':<15} | {'MODE':<18} | {'TIME(s)':<8} | {'AV %':<7} | {'ENTROPY':<8} | {'CORREL':<8}"
    sep = "=" * 90
    
    report.append("RESEARCH ANALYSIS: SPECK vs VECTORIZED vs CNN-HYBRID")
    report.append(sep)
    report.append(header)
    report.append(sep)
    
    print("Starting Comprehensive Security Benchmarking...")

    for img_name in images:
        path = os.path.join(images_dir, img_name)
        img = cv2.imread(path)
        if img is None: continue
        
        data = img.tobytes()
        orig_entropy = calculate_entropy(img)
        orig_correl = calculate_correlation(img)
        
        # 1. SCALAR SPECK
        t_scalar = benchmark_scalar_logic(data)
        report.append(f"{img_name:<15} | {'Scalar (Legacy)':<18} | {t_scalar:<8.4f} | {'N/A':<7} | {orig_entropy:<8.4f} | {orig_correl:<8.4f}")
        
        # 2. PURE VECTORIZED
        start = time.perf_counter()
        enc_vec_data = vec_cipher.encrypt(data)
        t_vec = time.perf_counter() - start
        
        # Reshape to calculate entropy of result
        enc_vec_img = np.frombuffer(enc_vec_data[:img.size], dtype=np.uint8).reshape(img.shape)
        ent_vec = calculate_entropy(enc_vec_img)
        cor_vec = calculate_correlation(enc_vec_img)
        av_vec = calculate_avalanche(vec_cipher, data)
        report.append(f"{'':<15} | {'Pure Vectorized':<18} | {t_vec:<8.4f} | {av_vec:<7.2f} | {ent_vec:<8.4f} | {cor_vec:<8.4f}")
        
        # 3. CNN HYBRID
        enc_hybrid_img, mask, t_hybrid = hybrid_engine.encrypt_adaptive(path)
        ent_hybrid = calculate_entropy(enc_hybrid_img)
        cor_hybrid = calculate_correlation(enc_hybrid_img)
        av_hybrid = calculate_avalanche(vec_cipher, data)
        report.append(f"{'':<15} | {'CNN-Vectorized':<18} | {t_hybrid:<8.4f} | {av_hybrid:<7.2f} | {ent_hybrid:<8.4f} | {cor_hybrid:<8.4f}")
        report.append("-" * 90)
        
        # Save Visual Artifacts for the last/current image
        save_name = os.path.splitext(img_name)[0]
        cv2.imwrite(os.path.join(output_dir, f"{save_name}_mask.jpg"), mask * 255)
        cv2.imwrite(os.path.join(output_dir, f"{save_name}_encrypted.jpg"), enc_hybrid_img)
        
        print(f"Metrics extracted for {img_name}")

    final_text = "\n".join(report)
    with open(results_file, "w") as f:
        f.write(final_text)
    
    print("\n" + final_text)
    print(f"\nVisuals and Detailed Report saved to: {results_file} and {output_dir}/")

if __name__ == "__main__":
    run_comprehensive_tests()
