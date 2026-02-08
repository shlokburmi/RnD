import cv2
import numpy as np
import matplotlib.pyplot as plt
import hashlib

def apply_sha512_to_image(image_path):
    """
    Load an image, generate a SHA512-based representation, and plot histograms.
    """
    # 1. Load the specific image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not load {image_path}")
        return

    print(f"Loaded {image_path} successfully.")
    print(f"Shape: {img.shape}")

    # 2. Calculate SHA512 Hash of the image content
    # We use the raw bytes of the image
    img_bytes = img.tobytes()
    sha512_hash = hashlib.sha512(img_bytes).hexdigest()
    print(f"\nSHA512 Hash: {sha512_hash}")

    # 3. Create a visual representation of the SHA512 hash
    # Convert hash hex string back to bytes
    hash_bytes = bytes.fromhex(sha512_hash)
    hash_array = np.frombuffer(hash_bytes, dtype=np.uint8)

    # We need to fill the same image dimensions with this hash pattern
    # The hash is only 64 bytes, so we repeat it to fill the image
    total_pixels = img.shape[0] * img.shape[1]
    
    # Repeat the hash bytes enough times to cover the image
    repetitions = (total_pixels // len(hash_array)) + 1
    encrypted_flat = np.tile(hash_array, repetitions)[:total_pixels]
    
    # Reshape to original image dimensions
    sha512_img = encrypted_flat.reshape(img.shape)

    # 4. Plot Histograms
    plt.figure(figsize=(15, 6))

    # Plot 1: Original Image
    plt.subplot(2, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Lenna (Grayscale)')
    plt.axis('off')

    # Plot 2: Original Histogram
    plt.subplot(2, 2, 2)
    plt.hist(img.ravel(), 256, [0, 256], color='black', alpha=0.7)
    plt.title('Histogram: Original')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)

    # Plot 3: SHA512 "Encrypted" Image
    plt.subplot(2, 2, 3)
    plt.imshow(sha512_img, cmap='gray')
    plt.title('SHA512 Representation')
    plt.axis('off')

    # Plot 4: SHA512 Histogram
    plt.subplot(2, 2, 4)
    plt.hist(sha512_img.ravel(), 256, [0, 256], color='red', alpha=0.7)
    plt.title('Histogram: SHA512')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = 'lenna_sha512_analysis.png'
    plt.savefig(output_file, dpi=300)
    print(f"\nAnalysis saved to: {output_file}")
    plt.show()

if __name__ == "__main__":
    apply_sha512_to_image('lenna_grayscale.webp')
