import cv2
import numpy as np
import matplotlib.pyplot as plt
import hashlib

def sha512_stream_encryption(image_path):
    """
    Encrypts an image using SHA-512 as a pseudo-random generator (Stream Cipher)
    to produce a ciphertext with a perfectly uniform histogram.
    """
    # 1. Load the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not load {image_path}")
        return

    height, width = img.shape
    total_pixels = height * width
    print(f"Processing Image: {width}x{height} pixels ({total_pixels} bytes)")

    # 2. Generate Keystream using SHA-512
    # We need a random byte for EVERY pixel to make it uniform.
    # SHA-512 gives 64 bytes at a time. We'll run it repeatedly with a counter.
    keystream_bytes = bytearray()
    
    # Use a seed key (in real usage, this would be a user password)
    key = b"SecretKey_For_Lenna_123" 
    
    # Calculate how many 64-byte blocks we need
    chunks_needed = (total_pixels // 64) + 1
    
    print("Generating keystream using SHA-512...")
    for i in range(chunks_needed):
        # Create a unique input for each block: Key + Counter
        # This ensures every block of the keystream is unique and random-looking
        data_to_hash = key + str(i).encode('utf-8')
        
        # Compute SHA-512 hash
        hash_digest = hashlib.sha512(data_to_hash).digest() # distinct 64 bytes
        keystream_bytes.extend(hash_digest)
    
    # Trim keystream to exact image size
    keystream = np.frombuffer(keystream_bytes[:total_pixels], dtype=np.uint8)
    keystream = keystream.reshape((height, width))
    
    # 3. Encrypt: XOR Image with Keystream
    # This diffuses the image pattern completely
    encrypted_img = cv2.bitwise_xor(img, keystream)

    # 4. Plot Histogram of the Encrypted Image
    plt.figure(figsize=(10, 6))
    
    # We use 256 bins for 0-255 pixel values
    plt.hist(encrypted_img.ravel(), bins=256, range=[0, 256], color='purple', alpha=0.7)
    
    plt.title('Histogram of SHA-512 Encrypted Image (Uniform Distribution)')
    plt.xlabel('Pixel Intensity (0-255)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.2)
    
    # Calculate Chi-square or entropy to prove uniformity (optional but cool)
    entropy = -np.sum((np.unique(encrypted_img, return_counts=True)[1] / total_pixels) * 
                      np.log2(np.unique(encrypted_img, return_counts=True)[1] / total_pixels))
    
    plt.text(0.02, 0.95, f'Entropy: {entropy:.4f} (Max: 8.0)', transform=plt.gca().transAxes, 
             fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    output_filename = 'lenna_sha512_uniform_histogram.png'
    plt.savefig(output_filename, dpi=300)
    print(f"Uniform histogram saved as '{output_filename}'")
    
    # Also save the encrypted image to see it looks like noise
    cv2.imwrite('lenna_encrypted_sha512.png', encrypted_img)
    print("Encrypted image saved as 'lenna_encrypted_sha512.png'")
    
    plt.show()

if __name__ == "__main__":
    sha512_stream_encryption('lenna_grayscale.webp')
