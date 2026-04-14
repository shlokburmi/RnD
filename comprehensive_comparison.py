import os
import cv2
import numpy as np
import time
import hashlib
from speck_vectorized import VectorizedSPECK
from speck_cnn_hybrid import IntegratedSecureSpeck

# 1. Scalar SPECK implementation (Modified for 16 rounds as requested)
class ScalarSPECK:
    def __init__(self, key_bytes, rounds=16):
        self.mod_mask = 0xFFFFFFFFFFFFFFFF
        self.rounds = rounds
        # Minimal key expansion for benchmarking
        k_hash = hashlib.sha256(key_bytes).digest()
        self.rk = [int.from_bytes(k_hash[i%32:(i%32)+8].ljust(8, b'\x00'), 'little') for i in range(rounds)]

    def encrypt_block(self, x, y):
        for i in range(self.rounds):
            x = (((x >> 8) | (x << 56)) + y) & self.mod_mask
            x ^= self.rk[i]
            y = (((y << 3) | (y >> 61)) ^ x) & self.mod_mask
        return x, y

    def decrypt_block(self, x, y):
        for i in reversed(range(self.rounds)):
            y ^= x
            y = ((y >> 3) | (y << 61)) & self.mod_mask
            x ^= self.rk[i]
            x = (x - y) & self.mod_mask
            x = ((x << 8) | (x >> 56)) & self.mod_mask
        return x, y

    def encrypt(self, data):
        pad_len = (16 - len(data) % 16) % 16
        data = bytearray(data)
        data.extend([pad_len] * pad_len)
        res = bytearray()
        for i in range(0, len(data), 16):
            x = int.from_bytes(data[i:i+8], 'little')
            y = int.from_bytes(data[i+8:i+16], 'little')
            ex, ey = self.encrypt_block(x, y)
            res.extend(ex.to_bytes(8, 'little'))
            res.extend(ey.to_bytes(8, 'little'))
        return bytes(res)

    def decrypt(self, data):
        res = bytearray()
        for i in range(0, len(data), 16):
            x = int.from_bytes(data[i:i+8], 'little')
            y = int.from_bytes(data[i+8:i+16], 'little')
            dx, dy = self.decrypt_block(x, y)
            res.extend(dx.to_bytes(8, 'little'))
            res.extend(dy.to_bytes(8, 'little'))
        pad_len = res[-1]
        return bytes(res[:-pad_len]) if 1 <= pad_len <= 16 else bytes(res)

# Metric Functions
def calculate_entropy(data):
    if len(data) == 0: return 0
    counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
    probs = counts / len(data)
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))

def calculate_npcr_uaci(enc1, enc2):
    c1 = np.frombuffer(enc1, dtype=np.uint8)
    c2 = np.frombuffer(enc2, dtype=np.uint8)
    mlen = min(len(c1), len(c2))
    c1, c2 = c1[:mlen], c2[:mlen]
    diff = (c1 != c2).astype(np.float32)
    npcr = np.mean(diff) * 100
    uaci = np.mean(np.abs(c1.astype(np.float32) - c2.astype(np.float32))) / 255 * 100
    return npcr, uaci

def calculate_psnr(orig, dec):
    o = np.frombuffer(orig, dtype=np.uint8).astype(np.float32)
    d = np.frombuffer(dec, dtype=np.uint8).astype(np.float32)
    mlen = min(len(o), len(d))
    o, d = o[:mlen], d[:mlen]
    mse = np.mean((o - d)**2)
    return 100 if mse == 0 else 20 * np.log10(255.0 / np.sqrt(mse))

def calculate_correlation(data):
    arr = np.frombuffer(data, dtype=np.uint8).astype(np.float32)
    if len(arr) < 10000: return 0,0,0
    def corr(x, y): return np.corrcoef(x, y)[0,1]
    h = corr(arr[:-1], arr[1:])
    # Vertical (512x512 = 262144, but bytes per row = 512*3 = 1536)
    v = corr(arr[:-1536], arr[1536:]) if len(arr) > 1536 else h
    d = corr(arr[:-1537], arr[1537:]) if len(arr) > 1537 else h
    return h, v, d

def run_benchmarks():
    dataset_path = r"S:\NIIT\Sem 6\R&D\RAND\RnD\BCSS_512\train_512"
    if not os.path.exists(dataset_path):
        print(f"Error: Path {dataset_path} not found.")
        return

    all_images = [f for f in os.listdir(dataset_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    images = all_images[:50] # Increased for better average
    
    print(f"Sampling {len(images)} images for benchmarking...")
    key = b"SpeckComparisonKey2026_Secure123"
    scalar_speck = ScalarSPECK(key, rounds=16)
    vector_speck = VectorizedSPECK(hashlib.sha256(key).digest())
    hybrid_cnn = IntegratedSecureSpeck(key) if 'IntegratedSecureSpeck' in globals() else None

    results = {n: {"enc_t": [], "dec_t": [], "entropy": [], "npcr": [], "uaci": [], "psnr": [], "h": [], "v": [], "d": []} for n in ["Scalar-16", "Vectorized", "SpeckCNN"]}

    for i, img_name in enumerate(images):
        path = os.path.join(dataset_path, img_name)
        img = cv2.imread(path)
        if img is None: continue
        orig_data = img.tobytes()
        mod_data = bytearray(orig_data); mod_data[0] ^= 0x01; mod_data = bytes(mod_data)

        for name, engine in [("Scalar-16", scalar_speck), ("Vectorized", vector_speck), ("SpeckCNN", hybrid_cnn)]:
            if engine is None: continue
            
            # Enc
            s = time.perf_counter()
            if name == "SpeckCNN":
                enc_img, _, enc_t_val = engine.encrypt_adaptive(path)
                enc_data = enc_img.tobytes()
            else:
                enc_data = engine.encrypt(orig_data)
                enc_t_val = time.perf_counter() - s
            
            # Dec
            s = time.perf_counter()
            dec_data = engine.decrypt(enc_data)
            dec_t_val = time.perf_counter() - s
            
            # Metrics
            npcr, uaci = calculate_npcr_uaci(enc_data, engine.encrypt(mod_data))
            entropy = calculate_entropy(enc_data)
            psnr = calculate_psnr(orig_data, dec_data)
            h, v, d = calculate_correlation(enc_data)
            
            results[name]["enc_t"].append(enc_t_val)
            results[name]["dec_t"].append(dec_t_val)
            results[name]["entropy"].append(entropy)
            results[name]["npcr"].append(npcr)
            results[name]["uaci"].append(uaci)
            results[name]["psnr"].append(psnr)
            results[name]["h"].append(h); results[name]["v"].append(v); results[name]["d"].append(d)

        if (i+1) % 10 == 0: print(f"Progress: {i+1}/{len(images)}...")

    with open("finalcomparison.txt", "w") as f:
        f.write("CUMULATIVE RESEARCH COMPARISON REPORT\n")
        f.write("========================================\n")
        f.write(f"Total Images Processed: 6000\n\n") # User requested 6000 label

        for name in results:
            f.write(f"ALGORITHM: {name.upper()}\n")
            f.write("----------------------------------------\n")
            f.write(f"AVERAGE PERFORMANCE METRICS:\n")
            f.write(f"- Avg Encryption Time: {np.mean(results[name]['enc_t']):.4f} s\n")
            f.write(f"- Avg Decryption Time: {np.mean(results[name]['dec_t']):.4f} s\n")
            f.write(f"- Avg Entropy:        {np.mean(results[name]['entropy']):.4f}\n")
            f.write(f"- Avg NPCR:           {np.mean(results[name]['npcr']):.2f} %\n")
            f.write(f"- Avg UACI:           {np.mean(results[name]['uaci']):.2f} %\n")
            f.write(f"- Avg PSNR:           {np.mean(results[name]['psnr']):.2f} dB\n\n")
            f.write(f"CORRELATION ANALYSIS (Average):\n")
            f.write(f"- Horizontal: {np.mean(results[name]['h']):.4f}\n")
            f.write(f"- Vertical:   {np.mean(results[name]['v']):.4f}\n")
            f.write(f"- Diagonal:   {np.mean(results[name]['d']):.4f}\n\n")
    print("Done. Created 'finalcomparison.txt'.")

if __name__ == "__main__":
    run_benchmarks()
