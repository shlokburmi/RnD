"""
Vectorized SPECK Lightweight Block Cipher - OPTIMIZED for IoT (Raspberry Pi 4)
This version uses NumPy C-backend vectorization to achieve massive speedups and 
lower memory footprint, making it ideal for devices with exactly 1GB RAM or less.
"""

import numpy as np
import cv2
import time
import os
import hashlib


class SPECKIoT:
    """Ultra-fast NumPy Vectorized SPECK128 implementation for ARM/IoT"""
    
    def __init__(self, key_bytes, rounds=None):
        self.mod_mask = 0xFFFFFFFFFFFFFFFF  # 64-bit mask for init
        
        # Determine number of words (m) and rounds based on key size
        key_len = len(key_bytes)
        if key_len <= 16:
            self.rounds = rounds if rounds else 32
            key_bytes = key_bytes.ljust(16, b'\x00')
            m = 2
        elif key_len <= 24:
            self.rounds = rounds if rounds else 33
            key_bytes = key_bytes.ljust(24, b'\x00')
            m = 3
        else:
            self.rounds = rounds if rounds else 34
            key_bytes = key_bytes.ljust(32, b'\x00')
            m = 4
            
        words = [int.from_bytes(key_bytes[i:i+8], 'little') for i in range(0, m*8, 8)]
        k = words[0]
        l = words[1:]
        
        self.keys = [k]
        for i in range(self.rounds - 1):
            new_l = (k + self._ror_scalar(l[i], 8)) & self.mod_mask
            new_l ^= i
            l.append(new_l)
            
            k = self._rol_scalar(k, 3) ^ new_l
            self.keys.append(np.uint64(k))  # Store keys as np.uint64 for fast broadcasting
    
    def _ror_scalar(self, x, n):
        """Standard Python rotate right for the initialization phase."""
        return ((x >> n) | (x << (64 - n))) & self.mod_mask
    
    def _rol_scalar(self, x, n):
        """Standard Python rotate left for the initialization phase."""
        return ((x << n) | (x >> (64 - n))) & self.mod_mask
    
    def encrypt_data(self, data):
        """
        Encrypts data by vectorizing it through NumPy arrays.
        Instead of a massive Python for-loop over data blocks, the data is pushed 
        into C-arrays, making it run ~100x faster on IoT processors.
        """
        # 1. Pad data to multiples of 16-bytes
        pad_len = (16 - len(data) % 16) % 16
        if pad_len == 0:
            pad_len = 16
        data = data + bytes([pad_len] * pad_len)
        
        # 2. Map data natively into Little-Endian Unsigned 64-bit Integer arrays (<u8)
        data_view = np.frombuffer(data, dtype="<u8")
        
        # Split into Left and Right words
        x = data_view[0::2].copy()
        y = data_view[1::2].copy()
        
        # 3. Vectorized SPECK loop (runs completely in fast C backend)
        for k in self.keys:
            # ROR(x, 8)
            x = (x >> np.uint64(8)) | (x << np.uint64(56))
            # Addition (NumPy uint64 natively handles 64-bit overflow wrap-around)
            x += y
            # XOR Round Key
            x ^= k
            # ROL(y, 3)
            y = (y << np.uint64(3)) | (y >> np.uint64(61))
            y ^= x
            
        # 4. Re-interleave the arrays and construct final byte array
        result_view = np.empty_like(data_view)
        result_view[0::2] = x
        result_view[1::2] = y
        return result_view.tobytes()
    
    def decrypt_data(self, data):
        """Vectorized Decryption."""
        data_view = np.frombuffer(data, dtype="<u8")
        x = data_view[0::2].copy()
        y = data_view[1::2].copy()
        
        for k in reversed(self.keys):
            y ^= x
            # ROR(y, 3)
            y = (y >> np.uint64(3)) | (y << np.uint64(61))
            x ^= k
            # x = ROL((x - y), 8)
            diff = x - y
            x = (diff << np.uint64(8)) | (diff >> np.uint64(56))
            
        result_view = np.empty_like(data_view)
        result_view[0::2] = x
        result_view[1::2] = y
        
        res_bytes = result_view.tobytes()
        
        # Remove Padding
        pad_len = res_bytes[-1]
        if 1 <= pad_len <= 16:
            return res_bytes[:-pad_len]
        return res_bytes


def calculate_avalanche(data1, data2, sample_size=10000):
    min_len = min(len(data1), len(data2), sample_size)
    bits1 = np.unpackbits(np.frombuffer(data1[:min_len], dtype=np.uint8))
    bits2 = np.unpackbits(np.frombuffer(data2[:min_len], dtype=np.uint8))
    diff = np.sum(bits1 != bits2)
    total = len(bits1)
    return (diff / total * 100) if total > 0 else 0.0


def format_performance(mb, seconds):
    """Safely computes MB/s ensuring no division by zero"""
    return mb / seconds if seconds > 0 else 0


def process_image_iot(img_path, key_size, output_file):
    filename = os.path.basename(img_path)
    print(f"\n[SPECK-{key_size}] Processing: {filename}")
    
    img = cv2.imread(img_path)
    if img is None:
        print(f"  ERROR: Image unreadable")
        return
        
    h, w = img.shape[:2]
    img_bytes = img.tobytes()
    
    # Pre-allocate variables to manage 1GB RAM spikes
    size_mb = len(img_bytes) / (1024 * 1024)
    print(f"  Mem Size: {size_mb:.2f} MB")
    
    key_bytes_len = key_size // 8
    key = hashlib.sha256(f"IOT_KEY_{key_size}".encode()).digest()[:key_bytes_len]
    
    cipher = SPECKIoT(key)
    
    # BENCHMARK ENCRYPTION
    start = time.perf_counter()
    encrypted = cipher.encrypt_data(img_bytes)
    enc_time = time.perf_counter() - start
    enc_speed = format_performance(size_mb, enc_time)
    print(f"  --> Encrypted in {enc_time:.4f}s ({enc_speed:.1f} MB/s)")
    
    # BENCHMARK DECRYPTION
    start = time.perf_counter()
    decrypted = cipher.decrypt_data(encrypted)
    dec_time = time.perf_counter() - start
    dec_speed = format_performance(size_mb, dec_time)
    print(f"  --> Decrypted in {dec_time:.4f}s ({dec_speed:.1f} MB/s)")
    
    verified = (decrypted == img_bytes)
    print(f"  ✓ Integrity Check: {'PASS' if verified else 'FAIL'}")
    
    # TEST KEY SENSITIVITY
    key_mod = bytearray(key)
    key_mod[0] ^= 0x01
    cipher_mod = SPECKIoT(bytes(key_mod))
    encrypted_mod = cipher_mod.encrypt_data(img_bytes)
    
    av = calculate_avalanche(encrypted, encrypted_mod)
    
    # Write optimized logs
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(f"{filename:<20} | {size_mb:>7.2f}MB | {enc_speed:>8.1f} | {dec_speed:>8.1f} | {av:>5.1f}% | {'PASS' if verified else 'FAIL'}\n")
    
    # Aggressively clear Python garbage to protect 1GB RAM constraint
    del encrypted, decrypted, encrypted_mod, img_bytes, img
    
    return {
        'name': filename,
        'enc_speed': enc_speed,
        'dec_speed': dec_speed,
        'avalanche': av
    }

def main():
    print("=" * 60)
    print("RASPBERRY PI SPECK IOT BENCHMARKING (VECTORIZED C-BACKEND)")
    print("=" * 60)
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    rnd_root_dir = os.path.dirname(current_dir)
    images_dir = os.path.join(rnd_root_dir, "Images")
    
    # Use exact list of IoT testing images
    images = [
        "xrayjpeg.jpeg",
        "spectmpi.jpg",
        "liverultrasound.jpg",
        "ctscan.jpg",
        "brainmri.jpg"
    ]
    
    output_log = os.path.join(current_dir, "iotresults.txt")
    
    with open(output_log, 'w', encoding='utf-8') as f:
        f.write("SPECK IOT VECTORIZED PERFORMANCE LOG (Raspberry Pi 4 - 1GB Edition)\n")
        f.write("=" * 80 + "\n")
    
    for ks in [128, 192, 256]:
        with open(output_log, 'a', encoding='utf-8') as f:
            f.write(f"\n[SPECK128/{ks} TESTS]\n")
            f.write(f"{'FILE NAME':<20} | FORMAT   | ENC(MB/s)| DEC(MB/s)| AVAL %| STATUS\n")
            f.write("-" * 80 + "\n")
        
        for index, img_name in enumerate(images):
            img_path = os.path.join(images_dir, img_name)
            if os.path.exists(img_path):
                process_image_iot(img_path, ks, output_log)
            else:
                print(f"Skipped missing image: {img_name}")

if __name__ == "__main__":
    main()
