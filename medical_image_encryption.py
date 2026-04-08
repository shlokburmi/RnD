import cv2
import numpy as np
import hashlib
import time
import os

def generate_keystream(total_bytes, key):
    """Generates a keystream using SHA-512."""
    keystream_bytes = bytearray()
    chunks_needed = (total_bytes // 64) + 1
    
    for i in range(chunks_needed):
        data_to_hash = key + str(i).encode('utf-8')
        hash_digest = hashlib.sha512(data_to_hash).digest()
        keystream_bytes.extend(hash_digest)
        
    return np.frombuffer(keystream_bytes[:total_bytes], dtype=np.uint8)

def process_image(image_path, results_file):
    filename = os.path.basename(image_path)
    
    print(f"\\nProcessing: {filename}")
    
    # Load Image
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    
    if img is None:
        msg = f"ERROR: Could not load {filename}\\n"
        print(msg)
        with open(results_file, "a", encoding='utf-8') as f:
            f.write(msg)
        return

    # Get dimensions
    if len(img.shape) == 3:
        height, width, channels = img.shape
    else:
        height, width = img.shape
        channels = 1
        
    total_bytes = img.nbytes
    size_mb = total_bytes / (1024 * 1024)
    
    print(f"  Dimensions: {width}x{height}, Channels: {channels}, Size: {size_mb:.4f} MB")

    key = b"Medical_Image_Secret_Key_2026"

    # ENCRYPTION
    start_enc = time.perf_counter()
    keystream_flat = generate_keystream(total_bytes, key)
    keystream = keystream_flat.reshape(img.shape)
    encrypted_img = cv2.bitwise_xor(img, keystream)
    end_enc = time.perf_counter()
    enc_time = end_enc - start_enc
    enc_speed = size_mb / enc_time if enc_time > 0 else 0

    print(f"  Encryption: {enc_time:.6f}s ({enc_speed:.2f} MB/s)")

    # DECRYPTION
    start_dec = time.perf_counter()
    keystream_flat_dec = generate_keystream(total_bytes, key)
    keystream_dec = keystream_flat_dec.reshape(encrypted_img.shape)
    decrypted_img = cv2.bitwise_xor(encrypted_img, keystream_dec)
    end_dec = time.perf_counter()
    dec_time = end_dec - start_dec
    dec_speed = size_mb / dec_time if dec_time > 0 else 0
    
    print(f"  Decryption: {dec_time:.6f}s ({dec_speed:.2f} MB/s)")
    
    # Verify
    is_same = np.array_equal(img, decrypted_img)
    verification_status = "SUCCESS" if is_same else "FAILED"
    print(f"  Verification: {verification_status}")

    # Write results
    with open(results_file, "a", encoding='utf-8') as f:
        f.write(f"Image: {filename}\\n")
        f.write(f"Dimensions: {width} x {height}\\n")
        f.write(f"Size: {size_mb:.4f} MB\\n\\n")
        f.write(f"Encryption Time: {enc_time:.6f} seconds\\n")
        f.write(f"Encryption Speed: {enc_speed:.2f} MB/s\\n\\n")
        f.write(f"Decryption Time: {dec_time:.6f} seconds\\n")
        f.write(f"Decryption Speed: {dec_speed:.2f} MB/s\\n")
        f.write(f"Verification: {verification_status}\\n")
        f.write("=" * 60 + "\\n\\n")

if __name__ == "__main__":
    print("SHA-512 Medical Image Encryption Analysis")
    print("=" * 60)
    
    images = [
        "ctscan.jpg",
        "brainmri.jpg", 
        "liverultrasound.jpg", 
        "xrayjpeg.jpeg", 
        "spectmpi.jpg"
    ]
    
    # Get current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    results_file = os.path.join(current_dir, "medicalimagesresults.txt")
    
    # Initialize results file
    with open(results_file, "w", encoding='utf-8') as f:
        f.write("Medical Image Encryption Results (SHA-512 Stream Cipher)\\n")
        f.write("=" * 60 + "\\n\\n")
        
    # Process each image
    for img_name in images:
        full_path = os.path.join(current_dir, img_name)
        process_image(full_path, results_file)
        
    print(f"\\n{'=' * 60}")
    print(f"Results saved to: {results_file}")
    print("Done!")
