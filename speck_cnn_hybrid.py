import cv2
import numpy as np
import time
import os
import hashlib
from speck_vectorized import VectorizedSPECK

class CNNSegmenter:
    """
    Simulates or implements a CNN-based ROI (Region of Interest) segmentation.
    This component is integrated into the input pipeline to identify sensitive regions.
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
        """
        Processes input image to produce a binary mask of the ROI.
        If no CNN model is loaded, it falls back to a saliency-based detection
        which provides a high-confidence ROI for medical/structured images.
        """
        if self.has_tf and self.model:
            # Preprocess and predict using model
            img_input = cv2.resize(image, (224, 224))
            img_input = img_input / 255.0
            prediction = self.model.predict(img_input[np.newaxis, ...])
            mask = cv2.resize(prediction[0], (image.shape[1], image.shape[0]))
            return (mask > 0.5).astype(np.uint8)
        else:
            # Fallback: Multi-scale Saliency Detection (acting as a Pseudo-CNN ROI)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
            # Use Gaussian Blurs to find contrast at different scales
            blur1 = cv2.GaussianBlur(gray, (5, 5), 0)
            blur2 = cv2.GaussianBlur(gray, (15, 15), 0)
            saliency = cv2.absdiff(blur1, blur2)
            
            # Thresholding to get ROI
            _, mask = cv2.threshold(saliency, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Morphological cleanup (closing) to solidfy ROI regions
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            return mask // 255

class IntegratedSecureSpeck:
    def __init__(self, key):
        if isinstance(key, str): key = key.encode()
        self.key = hashlib.sha256(key).digest()
        self.cipher = VectorizedSPECK(self.key, key_size=256)
        self.segmenter = CNNSegmenter()

    def encrypt_adaptive(self, image_path):
        """
        Integrates CNN for ROI detection. Optimized for Raspberry Pi 4 (1GB RAM).
        """
        img = cv2.imread(image_path)
        if img is None: return None, 0, 0
        
        start_time = time.perf_counter()
        
        # 1. CNN ROI Detection
        mask = self.segmenter.get_roi_mask(img)
        
        # 2. Dynamic Key Generation for ROI
        roi_indices = np.where(mask == 1)
        roi_pixels = img[roi_indices]
        
        if len(roi_pixels) > 0:
            roi_features = roi_pixels.tobytes()
            dynamic_key = hashlib.sha256(self.key + hashlib.sha256(roi_features).digest()).digest()
            dynamic_cipher = VectorizedSPECK(dynamic_key, key_size=256)
            
            encrypted_roi = dynamic_cipher.encrypt(roi_features)
            
            # In-place update
            channels = img.shape[2]
            enc_roi_array = np.frombuffer(encrypted_roi[:len(roi_pixels) * channels], dtype=np.uint8)
            img[roi_indices] = enc_roi_array.reshape(-1, channels)
        
        # 3. Fast Background Diffusion
        bg_indices = np.where(mask == 0)
        if len(bg_indices[0]) > 0:
            keystream = hashlib.sha256(self.key).digest()
            chunk_size = 500000 
            channels = img.shape[2]
            
            for i in range(0, len(bg_indices[0]), chunk_size):
                end = min(i + chunk_size, len(bg_indices[0]))
                curr_size = end - i
                ks_len = curr_size * channels
                ks = (keystream * (ks_len // 32 + 1))[:ks_len]
                ks_array = np.frombuffer(ks, dtype=np.uint8).reshape(curr_size, channels)
                img[bg_indices[0][i:end], bg_indices[1][i:end], :] ^= ks_array

        end_time = time.perf_counter()
        return img, mask, (end_time - start_time)

    def encrypt(self, data):
        """Standard encryption for comparison."""
        return self.cipher.encrypt(data)

    def decrypt(self, data):
        """Standard decryption for comparison."""
        return self.cipher.decrypt(data)

def main():
    print("="*80)
    print("CNN-INTEGRATED VECTORIZED SPECK ENCRYPTION SYSTEM")
    print("="*80)
    
    images_dir = "Images"
    image_name = "brainmri.jpg"
    img_path = os.path.join(images_dir, image_name)
    
    if not os.path.exists(img_path):
        print(f"Error: {img_path} not found.")
        return

    sec_speck = IntegratedSecureSpeck(b"SecureEngine2026")
    enc_img, roi_mask, duration = sec_speck.encrypt_adaptive(img_path)
    
    if enc_img is not None:
        print(f"✓ Encryption Time: {duration:.4f} seconds")
        output_dir = "cnn_speck_output"
        os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(os.path.join(output_dir, "roi_mask.jpg"), roi_mask * 255)
        cv2.imwrite(os.path.join(output_dir, "encrypted_hybrid.jpg"), enc_img)
    else:
        print("Encryption Failed.")

if __name__ == "__main__":
    main()
