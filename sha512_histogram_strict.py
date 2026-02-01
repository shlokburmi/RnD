import cv2
import numpy as np
import matplotlib.pyplot as plt
import hashlib

def sha512_histogram_only(image_path):
    """
    Applies SHA512 to the image and plots ONLY the histogram of the result.
    """
    # 1. Load image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not load {image_path}")
        return

    # 2. Compute SHA512 Hash
    img_bytes = img.tobytes()
    sha512_hash = hashlib.sha512(img_bytes).hexdigest()
    print(f"SHA512 Hash: {sha512_hash}")

    # 3. Create image representation from hash (Encryption/Visualization)
    # Convert hex hash to bytes
    hash_bytes = bytes.fromhex(sha512_hash)
    hash_array = np.frombuffer(hash_bytes, dtype=np.uint8)
    
    # Tile the hash to match original image size for visualization
    total_pixels = img.shape[0] * img.shape[1]
    repetitions = (total_pixels // len(hash_array)) + 1
    encrypted_flat = np.tile(hash_array, repetitions)[:total_pixels]
    
    # Reshape to image dimensions (results in the "Encrypted" image)
    encrypted_img = encrypted_flat.reshape(img.shape)

    # 4. Plot ONLY the Histogram of the computed SHA512 image
    plt.figure(figsize=(10, 6))
    
    # Use keyword arguments for range to avoid matplotlib warning
    plt.hist(encrypted_img.ravel(), bins=256, range=[0, 256], color='red', alpha=0.7)
    
    plt.title('Histogram of SHA512 Processed Image')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_filename = 'lenna_sha512_histogram_only.png'
    plt.savefig(output_filename, dpi=300)
    print(f"Histogram saved as '{output_filename}'")
    plt.show()

if __name__ == "__main__":
    sha512_histogram_only('lenna_grayscale.webp')
