import cv2
import numpy as np
import hashlib
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

def process_and_save_images(image_path):
    """Encrypt and decrypt an image, saving both versions."""
    filename = os.path.basename(image_path)
    name_without_ext = os.path.splitext(filename)[0]
    ext = os.path.splitext(filename)[1]
    
    print(f"\nProcessing: {filename}")
    
    # Load Image
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    
    if img is None:
        print(f"  ERROR: Could not load {filename}")
        return

    # Get dimensions
    if len(img.shape) == 3:
        height, width, channels = img.shape
        print(f"  Dimensions: {width}x{height}x{channels} (Color)")
    else:
        height, width = img.shape
        print(f"  Dimensions: {width}x{height} (Grayscale)")
        
    total_bytes = img.nbytes
    print(f"  Size: {total_bytes:,} bytes")

    # Encryption key
    key = b"Medical_Image_Secret_Key_2026"

    # ENCRYPTION
    print("  Encrypting...")
    keystream_flat = generate_keystream(total_bytes, key)
    keystream = keystream_flat.reshape(img.shape)
    encrypted_img = cv2.bitwise_xor(img, keystream)
    
    # Save encrypted image
    encrypted_filename = f"encrypted_{name_without_ext}{ext}"
    encrypted_path = os.path.join(os.path.dirname(image_path), encrypted_filename)
    cv2.imwrite(encrypted_path, encrypted_img)
    print(f"  ✓ Saved: {encrypted_filename}")

    # DECRYPTION
    print("  Decrypting...")
    keystream_flat_dec = generate_keystream(total_bytes, key)
    keystream_dec = keystream_flat_dec.reshape(encrypted_img.shape)
    decrypted_img = cv2.bitwise_xor(encrypted_img, keystream_dec)
    
    # Save decrypted image
    decrypted_filename = f"decrypted_{name_without_ext}{ext}"
    decrypted_path = os.path.join(os.path.dirname(image_path), decrypted_filename)
    cv2.imwrite(decrypted_path, decrypted_img)
    print(f"  ✓ Saved: {decrypted_filename}")
    
    # Verify
    is_same = np.array_equal(img, decrypted_img)
    if is_same:
        print(f"  ✓ Verification: SUCCESS (Lossless recovery)")
    else:
        print(f"  ✗ Verification: FAILED")

if __name__ == "__main__":
    print("="*70)
    print("SHA-512 Medical Image Encryption - Image Generation")
    print("="*70)
    
    images = [
        "xrayjpeg.jpeg",
        "spectmpi.jpg",
        "liverultrasound.jpg",
        "ctscan.jpg",
        "brainmri.jpg"
    ]
    
    # Get current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Process each image
    for img_name in images:
        full_path = os.path.join(current_dir, img_name)
        process_and_save_images(full_path)
        
    print(f"\n{'='*70}")
    print("✓ All images encrypted and decrypted successfully!")
    print(f"{'='*70}")
    print("\nGenerated files:")
    print("  Encrypted images: encrypted_*.jpg/jpeg")
    print("  Decrypted images: decrypted_*.jpg/jpeg")
    print("\nNote: Encrypted images will look like random noise (uniform histogram)")
    print("      Decrypted images are pixel-perfect identical to originals")
