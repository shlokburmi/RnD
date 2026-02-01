import numpy as np
from PIL import Image
import os
from cryptography.fernet import Fernet
import base64
import hashlib

class EncryptionAnalyzer:
    @staticmethod
    def calculate_npcr(img1_data, img2_data):
        """
        Calculate Number of Pixels Change Rate (NPCR).
        Measures the percentage of different pixel values between two images.
        Ideal value: ~99.6%
        """
        arr1 = np.frombuffer(img1_data, dtype=np.uint8)
        arr2 = np.frombuffer(img2_data, dtype=np.uint8)
        
        if len(arr1) != len(arr2):
            # Pad to match lengths for comparison if needed, or truncate
            min_len = min(len(arr1), len(arr2))
            arr1 = arr1[:min_len]
            arr2 = arr2[:min_len]
            
        diff_count = np.sum(arr1 != arr2)
        total_pixels = len(arr1)
        
        npcr = (diff_count / total_pixels) * 100
        return npcr

    @staticmethod
    def calculate_uaci(img1_data, img2_data):
        """
        Calculate Unified Average Changing Intensity (UACI).
        Measures the average intensity of differences between two images.
        Ideal value: ~33.46%
        """
        arr1 = np.frombuffer(img1_data, dtype=np.uint8)
        arr2 = np.frombuffer(img2_data, dtype=np.uint8)
        
        if len(arr1) != len(arr2):
            min_len = min(len(arr1), len(arr2))
            arr1 = arr1[:min_len]
            arr2 = arr2[:min_len]
            
        abs_diff = np.abs(arr1.astype(int) - arr2.astype(int))
        uaci = (np.sum(abs_diff) / (len(arr1) * 255)) * 100
        return uaci

    @staticmethod
    def calculate_avalanche_effect(data1, data2):
        """
        Calculate bit difference rate (Avalanche Effect).
        Ideal value: ~50%
        """
        # Convert bytes to bits
        bits1 = ''.join(format(byte, '08b') for byte in data1)
        bits2 = ''.join(format(byte, '08b') for byte in data2)
        
        if len(bits1) != len(bits2):
            min_len = min(len(bits1), len(bits2))
            bits1 = bits1[:min_len]
            bits2 = bits2[:min_len]
        
        diff_bits = sum(b1 != b2 for b1, b2 in zip(bits1, bits2))
        total_bits = len(bits1)
        
        return (diff_bits / total_bits) * 100

    @staticmethod
    def generate_difference_image(img1_data, img2_data, width, height, output_path):
        """
        Generate a visual representation of the difference between two encrypted images.
        Calculates |Img1 - Img2| and normalizes it.
        """
        arr1 = np.frombuffer(img1_data, dtype=np.uint8)
        arr2 = np.frombuffer(img2_data, dtype=np.uint8)
        
        # Ensure correct size for reshaping
        target_size = width * height * 3 # Assuming RGB
        if len(arr1) < target_size:
             # Basic padding if needed (though encryption usually adds padding)
             arr1 = np.pad(arr1, (0, target_size - len(arr1)), 'constant')
        if len(arr2) < target_size:
             arr2 = np.pad(arr2, (0, target_size - len(arr2)), 'constant')
             
        # Truncate if larger (e.g. padding from encryption) to fit image dimensions
        # Or just use the original image dimensions
        
        # For visualization, we'll try to reconstruct an image structure
        # If encrypted data is larger (due to padding/metadata), we'll slice it to image size
        display_arr1 = arr1[:target_size]
        display_arr2 = arr2[:target_size]
        
        diff = np.abs(display_arr1.astype(int) - display_arr2.astype(int)).astype(np.uint8)
        
        # Reshape to image
        try:
            diff_img_arr = diff.reshape((height, width, 3))
            img = Image.fromarray(diff_img_arr, 'RGB')
            img.save(output_path)
            return True
        except Exception as e:
            print(f"Error generating difference image: {e}")
            return False

class SimpleEncryptor:
    @staticmethod
    def get_random_key():
        return Fernet.generate_key()
    
    @staticmethod
    def get_modified_key(original_key):
        """Flip the last bit of the key to generate a slightly different key."""
        # Decode base64 to bytes
        key_bytes = bytearray(base64.urlsafe_b64decode(original_key))
        # Flip the last bit of the last byte
        key_bytes[-1] ^= 1
        # Re-encode
        return base64.urlsafe_b64encode(key_bytes)
        
    @staticmethod
    def encrypt(data, key):
        f = Fernet(key)
        return f.encrypt(data)
