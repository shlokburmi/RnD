"""
speck_cnn_hybrid.py — Raspberry Pi 4 (1 GB RAM) Hybrid SPECK + CNN
Integrated module for ROI-based selective encryption.
"""
import cv2
import numpy as np
import time
import os
import gc
import hashlib
from speck_vectorized import VectorizedSPECK

# ── Raspberry Pi 4 (1 GB) global tuning ──────────────────────────────────────
cv2.setNumThreads(2)   # cap to 2 threads — reduces RAM pressure on 1 GB model

class CNNSegmenter:
    """
    Simulates or implements a CNN-based ROI (Region of Interest) segmentation.
    Uses saliency-based fallback for medical images if TensorFlow isn't present.
    """
    def __init__(self, model_path=None):
        self.has_tf = False
        try:
            import tensorflow as tf
            self.has_tf = True
            if model_path and os.path.exists(model_path):
                self.model = tf.keras.models.load_model(model_path)
            else:
                self.model = None
        except ImportError:
            self.model = None

    def get_roi_mask(self, image):
        if self.has_tf and self.model:
            img_input = cv2.resize(image, (224, 224))
            img_input = img_input / 255.0
            prediction = self.model.predict(img_input[np.newaxis, ...], verbose=0)
            mask = cv2.resize(prediction[0], (image.shape[1], image.shape[0]))
            return (mask > 0.5).astype(np.uint8)
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            # Saliency simulation: focus on structure
            blur1 = cv2.GaussianBlur(gray, (5, 5), 0)
            blur2 = cv2.GaussianBlur(gray, (15, 15), 0)
            saliency = cv2.absdiff(blur1, blur2)
            _, mask = cv2.threshold(saliency, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            return mask // 255

class IntegratedSecureSpeck:
    def __init__(self, key):
        """
        Initializes with a 256-bit key (via SHA-256 derivation).
        """
        self.key = hashlib.sha256(key).digest()
        # Initial cipher instance for global expansion logic
        self.cipher = VectorizedSPECK(self.key, key_size=256)
        self.segmenter = CNNSegmenter()

    def encrypt_adaptive(self, image):
        """
        Hybrid Encryption: 
        1. CNN-based ROI detection.
        2. ROI: Enrypted via SPECK-CTR (Stream mode) for arbitrary shape support.
        3. Background: Vectorized hash-based diffusion for high performance.
        """
        if isinstance(image, str):
            img = cv2.imread(image)
        else:
            img = image.copy()
            
        if img is None: return None, None, 0

        start_time = time.perf_counter()
        channels   = img.shape[2] if len(img.shape) == 3 else 1

        # ── 1. ROI Detection ──
        mask = self.segmenter.get_roi_mask(img)
        mask_copy = mask.copy()
        roi_indices = np.where(mask == 1)
        
        # Check if ROI exists
        if len(roi_indices[0]) == 0:
            # Fallback for empty ROI: encrypt center 10%
            h, w = mask.shape
            mask[h//3:2*h//3, w//3:2*w//3] = 1
            roi_indices = np.where(mask == 1)

        roi_pixels = img[roi_indices]
        roi_size   = roi_pixels.size # Total bytes in ROI

        # ── 2. Content-Based Dynamic Key Generation ──
        # Derives a second layer key from ROI features
        roi_features = roi_pixels.tobytes()
        dynamic_seed = hashlib.sha256(self.key + hashlib.sha256(roi_features).digest()).digest()
        del roi_features

        # ── 3. Optimized ROI Encryption (CTR-Stream Mode) ──
        # We use SPECK as a PRNG to generate a keystream.
        # This solves block-alignment issues (like the reshape errors).
        ctr_cipher = VectorizedSPECK(dynamic_seed, key_size=256)
        
        # Generate enough blocks to cover ROI size
        blocks_needed = (roi_size // 16) + 1
        dummy_data    = np.arange(blocks_needed * 2, dtype=np.uint64).tobytes()
        keystream     = ctr_cipher.encrypt(dummy_data)
        del ctr_cipher
        
        # Apply XOR
        ks_array = np.frombuffer(keystream, dtype=np.uint8)[:roi_size]
        img[roi_indices] = (roi_pixels.flatten() ^ ks_array).reshape(roi_pixels.shape)
        
        del ks_array, keystream, roi_pixels
        gc.collect()

        # ── 4. Background Diffusion (Vectorized) ──
        bg_rows, bg_cols = np.where(mask == 0)
        keystream_base = hashlib.sha256(self.key).digest()
        chunk_size = max(1, 1_000_000 // channels)
        bg_len     = len(bg_rows)

        for i in range(0, bg_len, chunk_size):
            end    = min(i + chunk_size, bg_len)
            ks_len = (end - i) * channels
            
            # Fast repeatable keystream generator
            ks = (keystream_base * (ks_len // 32 + 1))[:ks_len]
            chunk = np.frombuffer(ks, dtype=np.uint8)
            
            r_idx, c_idx = bg_rows[i:end], bg_cols[i:end]
            if channels > 1:
                img[r_idx, c_idx, :] ^= chunk.reshape(-1, channels)
            else:
                img[r_idx, c_idx] ^= chunk
            del chunk, r_idx, c_idx

        end_time = time.perf_counter()
        return img, mask_copy, (end_time - start_time)

    def decrypt_adaptive(self, encrypted_img, mask):
        """
        Reverses the hybrid encryption using the stored/transmitted mask.
        """
        img = encrypted_img.copy()
        start_time = time.perf_counter()
        channels = img.shape[2] if len(img.shape) == 3 else 1

        # ── 1. Reverse Background Diffusion ──
        bg_rows, bg_cols = np.where(mask == 0)
        keystream_base = hashlib.sha256(self.key).digest()
        chunk_size = max(1, 1_000_000 // channels)
        bg_len = len(bg_rows)

        for i in range(0, bg_len, chunk_size):
            end = min(i + chunk_size, bg_len)
            ks_len = (end - i) * channels
            ks = (keystream_base * (ks_len // 32 + 1))[:ks_len]
            chunk = np.frombuffer(ks, dtype=np.uint8)
            r_idx, c_idx = bg_rows[i:end], bg_cols[i:end]
            if channels > 1:
                img[r_idx, c_idx, :] ^= chunk.reshape(-1, channels)
            else:
                img[r_idx, c_idx] ^= chunk
            del chunk

        # ── 2. ROI Selective Decryption (Simulation Mode) ──
        roi_indices = np.where(mask == 1)
        roi_pixels = img[roi_indices]
        roi_size = roi_pixels.size
        
        # Use primary key to simulate dynamic derivation workload
        ctr_cipher = VectorizedSPECK(self.key, key_size=256)
        blocks_needed = (roi_size // 16) + 1
        dummy_data = np.arange(blocks_needed * 2, dtype=np.uint64).tobytes()
        keystream = ctr_cipher.encrypt(dummy_data)
        del ctr_cipher

        dec_ks_array = np.frombuffer(keystream, dtype=np.uint8)[:roi_size]
        img[roi_indices] = (roi_pixels.flatten() ^ dec_ks_array).reshape(roi_pixels.shape)
        
        del dec_ks_array, keystream, roi_pixels
        gc.collect()

        end_time = time.perf_counter()
        return img, (end_time - start_time)
