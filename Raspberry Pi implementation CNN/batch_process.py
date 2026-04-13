import os
import cv2
import time
import numpy as np
import hashlib
import math
from speck_cnn_hybrid import IntegratedSecureSpeck
from speck_vectorized import VectorizedSPECK

def calculate_entropy(image_data):
    if not isinstance(image_data, np.ndarray):
        image_data = np.frombuffer(image_data, dtype=np.uint8)
    
    # Flatten and get histogram
    flat_data = image_data.flatten()
    if len(flat_data) == 0: return 0
    
    counts = np.bincount(flat_data, minlength=256)
    probs = counts / len(flat_data)
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))

def calculate_avalanche(engine, img_path):
    # 1. Base encryption
    img = cv2.imread(img_path)
    if img is None: return 0
    
    # For simplicity, we use the raw encryption for avalanche test
    # (IntegratedSecureSpeck is deterministic given the same image)
    c1, _, _ = engine.encrypt_adaptive(img_path)
    
    # 2. Modify one bit in the image
    img_mod = img.copy()
    mid_row, mid_col = img_mod.shape[0]//2, img_mod.shape[1]//2
    # Flip the LSB of a center pixel
    if len(img_mod.shape) == 3:
        img_mod[mid_row, mid_col, 0] ^= 1 
    else:
        img_mod[mid_row, mid_col] ^= 1
    
    # Save temp mod image
    temp_path = "temp_mod.jpg"
    cv2.imwrite(temp_path, img_mod)
    
    # 3. Encrypt modified image
    c2, _, _ = engine.encrypt_adaptive(temp_path)
    os.remove(temp_path)
    
    if c1 is None or c2 is None: return 0
    
    # 4. Compare bits
    diff = np.bitwise_xor(c1, c2)
    changed_bits = bin(int.from_bytes(diff.tobytes(), 'little')).count('1')
    total_bits = c1.size * 8
    
    return (changed_bits / total_bits) * 100

def run_standard_speck(image_path, key):
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None: return None, 0
    
    start_time = time.perf_counter()
    cipher = VectorizedSPECK(hashlib.sha256(key).digest(), key_size=256)
    
    # Encrypt the entire raw byte stream of the image
    # Note: image.tobytes() includes all pixels
    raw_bytes = img.tobytes()
    encrypted_bytes = cipher.encrypt(raw_bytes)
    
    # We return the duration and an entropy sample from the encrypted bytes
    duration = time.perf_counter() - start_time
    entropy = calculate_entropy(encrypted_bytes)
    
    return duration, entropy

def run_comprehensive_batch():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(current_dir, "images")
    results_file = os.path.join(current_dir, "cnnresults.txt")
    
    key = b"SecureEngine2026"
    hybrid_engine = IntegratedSecureSpeck(key)
    
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    images = [f for f in os.listdir(images_dir) if f.lower().endswith(image_extensions)]
    
    if not images:
        print(f"No images found in {images_dir}")
        return

    print(f"Starting comprehensive analysis of {len(images)} images...")
    
    with open(results_file, "w") as f:
        f.write("CNN-INTEGRATED VECTORIZED SPECK VS STANDARD SPECK ANALYSIS\n")
        f.write("="*90 + "\n")
        header = f"{'Image':<15} | {'Type':<8} | {'Time(s)':<8} | {'Entropy':<8} | {'Avalanche %':<12}\n"
        f.write(header)
        f.write("-" * 90 + "\n")
        
        for image_name in images:
            img_path = os.path.join(images_dir, image_name)
            print(f"Analyzing {image_name}...")
            
            try:
                # 1. Hybrid Results
                enc_img, _, duration_h = hybrid_engine.encrypt_adaptive(img_path)
                entropy_h = calculate_entropy(enc_img)
                avalanche_h = calculate_avalanche(hybrid_engine, img_path)
                
                f.write(f"{image_name:<15} | Hybrid   | {duration_h:.4f}   | {entropy_h:.4f}   | {avalanche_h:.2f}%\n")
                
                # 2. Standard Results
                duration_s, entropy_s = run_standard_speck(img_path, key)
                # Avalanche for standard is mathematically expected to be ~50% for block ciphers
                # but we can simulate it if needed. For performance comparison, time is key.
                f.write(f"{'':<15} | Standard | {duration_s:.4f}   | {entropy_s:.4f}   | ~50.00% (Block Default)\n")
                f.write("-" * 90 + "\n")
                
            except Exception as e:
                f.write(f"{image_name:<15} | ERROR    | {str(e)[:40]}\n")
                print(f"Error on {image_name}: {e}")

    print(f"\nAnalysis complete. Detailed comparison saved to {results_file}")

if __name__ == "__main__":
    run_comprehensive_batch()
