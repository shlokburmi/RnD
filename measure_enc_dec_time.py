import cv2
import numpy as np
import hashlib
import time
import os

def get_keystream(height, width, key_bytes):
    """
    Generates the SHA-512 keystream.
    Separate function to simulate real-world overhead if needed, 
    but for stream ciphers, keystream generation is part of the process.
    """
    total_pixels = height * width
    keystream_bytes = bytearray()
    
    # Calculate chunks
    chunks_needed = (total_pixels // 64) + 1
    
    for i in range(chunks_needed):
        # Key + Counter strategy
        data_to_hash = key_bytes + str(i).encode('utf-8')
        hash_digest = hashlib.sha512(data_to_hash).digest()
        keystream_bytes.extend(hash_digest)
    
    keystream = np.frombuffer(keystream_bytes[:total_pixels], dtype=np.uint8)
    return keystream.reshape((height, width))

def sha512_crypt(image, key_bytes):
    """
    Encrypts/Decrypts image using SHA-512 stream cipher logic.
    Since it's XOR, Encryption and Decryption are the same operation.
    """
    height, width = image.shape
    keystream = get_keystream(height, width, key_bytes)
    return cv2.bitwise_xor(image, keystream)

def main():
    input_path = "lenna_grayscale.webp"
    output_txt = "encrdecr.txt"
    decrypted_img_path = "lenna_decrypted_sha512.png"
    
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return

    # 1. Load Image
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not read {input_path}")
        return
        
    height, width = img.shape
    size_mb = (height * width) / (1024 * 1024)
    key = b"SecretKey_For_Time_Test"
    
    print(f"Processing {input_path} ({width}x{height} pixels)...")

    # 2. Measure Encryption Time
    start_enc = time.perf_counter()
    encrypted_img = sha512_crypt(img, key)
    end_enc = time.perf_counter()
    encryption_time = end_enc - start_enc
    
    print(f"Encryption Time: {encryption_time:.6f} seconds")

    # 3. Measure Decryption Time
    # (Decryption is the same XOR operation with the same key)
    start_dec = time.perf_counter()
    decrypted_img = sha512_crypt(encrypted_img, key)
    end_dec = time.perf_counter()
    decryption_time = end_dec - start_dec
    
    print(f"Decryption Time: {decryption_time:.6f} seconds")

    # 4. Verification
    # Check if decrypted matches original
    difference = cv2.absdiff(img, decrypted_img)
    if np.count_nonzero(difference) == 0:
        status = "SUCCESS (Lossless Decryption)"
    else:
        status = "FAILED (Decryption Mismatch)"
    
    print(f"Decryption Status: {status}")

    # 5. Save Results
    # Also save the decrypted image to prove it works
    cv2.imwrite(decrypted_img_path, decrypted_img)
    
    result_text = (
        "SHA-512 Stream Cipher Performance Analysis\n"
        "==========================================\n"
        f"Image: {input_path}\n"
        f"Dimensions: {width} x {height}\n"
        f"Size: {size_mb:.4f} MB\n\n"
        f"Encryption Time: {encryption_time:.6f} seconds\n"
        f"Encryption Speed: {size_mb/encryption_time:.2f} MB/s\n\n"
        f"Decryption Time: {decryption_time:.6f} seconds\n"
        f"Decryption Speed: {size_mb/decryption_time:.2f} MB/s\n\n"
        f"Verification: {status}\n"
    )
    
    with open(output_txt, "w") as f:
        f.write(result_text)
        
    print(f"\nResults saved to {output_txt}")

if __name__ == "__main__":
    main()
