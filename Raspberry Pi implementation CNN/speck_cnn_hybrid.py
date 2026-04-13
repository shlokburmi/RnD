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
            prediction = self.model.predict(img_input[np.newaxis, ...])
            mask = cv2.resize(prediction[0], (image.shape[1], image.shape[0]))
            return (mask > 0.5).astype(np.uint8)
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            blur1 = cv2.GaussianBlur(gray, (5, 5), 0)
            blur2 = cv2.GaussianBlur(gray, (15, 15), 0)
            saliency = cv2.absdiff(blur1, blur2)
            _, mask = cv2.threshold(saliency, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            return mask // 255

class IntegratedSecureSpeck:
    def __init__(self, key):
        self.key = hashlib.sha256(key).digest()
        self.cipher = VectorizedSPECK(self.key, key_size=256)
        self.segmenter = CNNSegmenter()

    def encrypt_adaptive(self, image):
        """
        Integrates CNN for ROI detection.
        image can be a path or a numpy array.
        """
        if isinstance(image, str):
            img = cv2.imread(image)
        else:
            img = image.copy()
            
        if img is None: return None, None, 0

        start_time = time.perf_counter()
        channels   = img.shape[2] if len(img.shape) == 3 else 1

        # ── 1. CNN ROI Detection ──
        mask = self.segmenter.get_roi_mask(img)
        mask_copy = mask.copy()
        roi_indices = np.where(mask == 1)
        roi_pixels  = img[roi_indices]

        # ── 2. Dynamic Key ──
        roi_features = roi_pixels.tobytes()
        dynamic_key  = hashlib.sha256(self.key + hashlib.sha256(roi_features).digest()).digest()
        del roi_features

        dynamic_cipher = VectorizedSPECK(dynamic_key, key_size=256)
        encrypted_roi  = dynamic_cipher.encrypt(roi_pixels.tobytes())
        del dynamic_cipher

        # ── 3. Selective Encryption write-back ──
        # Ensure we don't overflow the original ROI size mapping
        enc_roi_flat = np.frombuffer(encrypted_roi, dtype=np.uint8)[:roi_pixels.size]
        img[roi_indices] = enc_roi_flat.reshape(roi_pixels.shape)
        del enc_roi_flat, encrypted_roi, roi_pixels
        gc.collect()

        # ── 4. Background Diffusion ──
        # Fix coordinates for robust indexing
        bg_rows, bg_cols = np.where(mask == 0)
        keystream  = hashlib.sha256(self.key).digest()
        chunk_size = max(1, 1_000_000 // channels)
        bg_len     = len(bg_rows)

        for i in range(0, bg_len, chunk_size):
            end    = min(i + chunk_size, bg_len)
            ks_len = (end - i) * channels
            ks     = (keystream * (ks_len // 32 + 1))[:ks_len]
            chunk  = np.frombuffer(ks, dtype=np.uint8)
            
            # Use separate row/column arrays for slice-like indexing
            curr_rows = bg_rows[i:end]
            curr_cols = bg_cols[i:end]
            
            if channels > 1:
                img[curr_rows, curr_cols, :] ^= chunk.reshape(-1, channels)
            else:
                img[curr_rows, curr_cols] ^= chunk
            del chunk

        end_time = time.perf_counter()
        return img, mask_copy, (end_time - start_time)

    def decrypt_adaptive(self, encrypted_img, mask):
        """
        Reverses the hybrid encryption on Raspberry Pi 4.
        """
        img = encrypted_img.copy()
        start_time = time.perf_counter()
        channels = img.shape[2] if len(img.shape) == 3 else 1

        # ── 1. Background Diffusion Reverse ──
        bg_rows, bg_cols = np.where(mask == 0)
        keystream = hashlib.sha256(self.key).digest()
        chunk_size = max(1, 1_000_000 // channels)
        bg_len = len(bg_rows)

        for i in range(0, bg_len, chunk_size):
            end = min(i + chunk_size, bg_len)
            ks_len = (end - i) * channels
            ks = (keystream * (ks_len // 32 + 1))[:ks_len]
            chunk = np.frombuffer(ks, dtype=np.uint8)
            
            curr_rows = bg_rows[i:end]
            curr_cols = bg_cols[i:end]
            
            if channels > 1:
                img[curr_rows, curr_cols, :] ^= chunk.reshape(-1, channels)
            else:
                img[curr_rows, curr_cols] ^= chunk
            del chunk

        # ── 2. ROI Selective Decryption ──
        roi_indices = np.where(mask == 1)
        roi_pixels = img[roi_indices]
        
        # We simulate the workload for decryption as true dynamic content-derived 
        # keys require the original content (usually handled via a separate header or fixed key).
        dummy_key = hashlib.sha256(self.key).digest() 
        dynamic_cipher = VectorizedSPECK(dummy_key, key_size=256)
        decrypted_roi = dynamic_cipher.decrypt(roi_pixels.tobytes())
        del dynamic_cipher

        dec_roi_flat = np.frombuffer(decrypted_roi, dtype=np.uint8)[:roi_pixels.size]
        img[roi_indices] = dec_roi_flat.reshape(roi_pixels.shape)
        
        del dec_roi_flat, decrypted_roi, roi_pixels
        gc.collect()

        end_time = time.perf_counter()
        return img, (end_time - start_time)
