import os
from PIL import Image
import numpy as np
from encryption_sensitivity import EncryptionAnalyzer, SimpleEncryptor

def modify_one_pixel(image_path):
    """Load image and return data with 1 pixel modified."""
    img = Image.open(image_path).convert('RGB')
    img_data = np.array(img).copy()
    print(f"DEBUG: img_path={image_path}, shape={img_data.shape}, dtype={img_data.dtype}")
    
    # Modify the first pixel very slightly (e.g., +1 value)
    # Since we converted to RGB, we know it has 3 channels
    try:
        original_val = img_data[0, 0, 0]
        # Cast to int to avoid overflow/numpy issues
        new_val = (int(original_val) + 1) % 256
        img_data[0, 0, 0] = new_val
    except Exception as e:
        print(f"CRITICAL ERROR in modify_one_pixel: {e}")
        import traceback
        traceback.print_exc()
        # Fallback to return original image to avoid crash
        return img
        
    return Image.fromarray(img_data)

def run_tests():
    image_path = "lenna_padded.webp"
    if not os.path.exists(image_path):
        # Fallback if padded version doesn't exist
        image_path = "lenna.webp"

    print(f"Running Encryption Analysis on: {image_path}")
    
    # Load Image
    img = Image.open(image_path).convert('RGB')
    print(f"Image Mode: {img.mode}, Size: {img.size}")
    
    width, height = img.size
    img_bytes = img.tobytes()

    
    # ==========================================
    # 1. KEY SENSITIVITY TEST
    # Exerimental setup: Same Plaintext, Slightly Different Keys
    # ==========================================
    print("\n" + "="*50)
    print("TEST 1: KEY SENSITIVITY (Avalanche Effect)")
    print("Condition: Same Image, Keys differ by 1 bit")
    print("="*50)
    
    key1 = SimpleEncryptor.get_random_key()
    key2 = SimpleEncryptor.get_modified_key(key1)
    
    print("Encrypting with Key 1...")
    cipher1 = SimpleEncryptor.encrypt(img_bytes, key1)
    print("Encrypting with Key 2...")
    cipher2 = SimpleEncryptor.encrypt(img_bytes, key2)
    
    # Analyze
    npcr = EncryptionAnalyzer.calculate_npcr(cipher1, cipher2)
    uaci = EncryptionAnalyzer.calculate_uaci(cipher1, cipher2)
    avalanche = EncryptionAnalyzer.calculate_avalanche_effect(cipher1, cipher2)
    
    print(f"\nResults for Key Sensitivity:")
    print(f"NPCR (Number of Pixels Change Rate): {npcr:.4f}% (Ideal: ~99.6%)")
    print(f"UACI (Unified Average Changing Intensity): {uaci:.4f}% (Ideal: ~33.4%)")
    print(f"Avalanche Effect (Bit Change Rate): {avalanche:.4f}% (Ideal: ~50%)")
    
    pass_key = avalanche >= 50.0  # Strict check might fail due to randomness, usually around 50%
    print(f"Status: {'STRONG' if avalanche > 48 else 'WEAK'} sensitivity")
    
    # Visualize Difference
    diff_file = "diff_key_sensitivity.png"
    EncryptionAnalyzer.generate_difference_image(cipher1, cipher2, width, height, diff_file)
    print(f"Generated difference image: {diff_file}")

    # ==========================================
    # 2. PLAINTEXT SENSITIVITY TEST
    # Experimental setup: Slightly Different Plaintext, Same Key
    # ==========================================
    print("\n" + "="*50)
    print("TEST 2: PLAINTEXT SENSITIVITY")
    print("Condition: Images differ by 1 pixel, Same Key")
    print("="*50)
    
    # Create modified image
    mod_img = modify_one_pixel(image_path)
    mod_img_bytes = mod_img.tobytes()
    
    # Encrypt both with same key
    key_shared = SimpleEncryptor.get_random_key()
    
    print("Encrypting Original Image...")
    cipher_p1 = SimpleEncryptor.encrypt(img_bytes, key_shared)
    print("Encrypting Modified Image (1 pixel diff)...")
    cipher_p2 = SimpleEncryptor.encrypt(mod_img_bytes, key_shared)
    
    # Analyze
    npcr_p = EncryptionAnalyzer.calculate_npcr(cipher_p1, cipher_p2)
    uaci_p = EncryptionAnalyzer.calculate_uaci(cipher_p1, cipher_p2)
    avalanche_p = EncryptionAnalyzer.calculate_avalanche_effect(cipher_p1, cipher_p2)
    
    print(f"\nResults for Plaintext Sensitivity:")
    print(f"NPCR: {npcr_p:.4f}%")
    print(f"UACI: {uaci_p:.4f}%")
    print(f"Avalanche Effect: {avalanche_p:.4f}%")
    
    # Visualize Difference
    diff_p_file = "diff_plaintext_sensitivity.png"
    EncryptionAnalyzer.generate_difference_image(cipher_p1, cipher_p2, width, height, diff_p_file)
    print(f"Generated difference image: {diff_p_file}")

if __name__ == "__main__":
    run_tests()
