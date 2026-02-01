import cv2
import numpy as np
import matplotlib.pyplot as plt
import hashlib
from PIL import Image

def load_lenna_grayscale():
    """
    Load Lenna image in grayscale.
    If not available locally, download it.
    """
    try:
        # Try to load from local file
        img = cv2.imread('lenna.png', cv2.IMREAD_GRAYSCALE)
        if img is None:
            # If not found, try to download
            print("Lenna image not found locally. Attempting to download...")
            import urllib.request
            url = 'https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png'
            urllib.request.urlretrieve(url, 'lenna.png')
            img = cv2.imread('lenna.png', cv2.IMREAD_GRAYSCALE)
        return img
    except Exception as e:
        print(f"Error loading image: {e}")
        # Create a sample grayscale image if download fails
        print("Creating a sample grayscale image instead...")
        return np.random.randint(0, 256, (512, 512), dtype=np.uint8)

def plot_original_histogram(image):
    """
    Plot histogram of the original grayscale image.
    """
    plt.figure(figsize=(12, 5))
    
    # Display the image
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Lenna Grayscale Image')
    plt.axis('off')
    
    # Plot histogram
    plt.subplot(1, 2, 2)
    histogram = cv2.calcHist([image], [0], None, [256], [0, 256])
    plt.plot(histogram, color='black')
    plt.title('Histogram of Original Image')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.xlim([0, 256])
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('original_histogram.png', dpi=300, bbox_inches='tight')
    print("Original histogram saved as 'original_histogram.png'")
    plt.show()

def apply_sha512_and_plot(image):
    """
    Apply SHA512 algorithm to the image and plot the resulting histogram.
    The SHA512 hash is used to generate a new image representation.
    """
    # Convert image to bytes
    image_bytes = image.tobytes()
    
    # Calculate SHA512 hash
    sha512_hash = hashlib.sha512(image_bytes).hexdigest()
    print(f"\nSHA512 Hash of the image: {sha512_hash}")
    
    # Convert hash to bytes and create a new image representation
    # We'll use the hash to generate pixel values
    hash_bytes = bytes.fromhex(sha512_hash)
    
    # Create a new image from hash bytes
    # SHA512 produces 64 bytes (512 bits), so we'll tile it to match original image size
    hash_array = np.frombuffer(hash_bytes, dtype=np.uint8)
    
    # Tile the hash to create an image of the same size as original
    rows, cols = image.shape
    total_pixels = rows * cols
    
    # Repeat hash bytes to fill the image
    repetitions = (total_pixels // len(hash_array)) + 1
    hash_image = np.tile(hash_array, repetitions)[:total_pixels]
    hash_image = hash_image.reshape(rows, cols)
    
    # Plot the SHA512-based image and its histogram
    plt.figure(figsize=(12, 5))
    
    # Display the SHA512-based image
    plt.subplot(1, 2, 1)
    plt.imshow(hash_image, cmap='gray')
    plt.title('SHA512-Based Image Representation')
    plt.axis('off')
    
    # Plot histogram of SHA512-based image
    plt.subplot(1, 2, 2)
    histogram = cv2.calcHist([hash_image], [0], None, [256], [0, 256])
    plt.plot(histogram, color='red')
    plt.title('Histogram of SHA512-Based Image')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.xlim([0, 256])
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('sha512_histogram.png', dpi=300, bbox_inches='tight')
    print("SHA512 histogram saved as 'sha512_histogram.png'")
    plt.show()
    
    return hash_image, sha512_hash

def compare_histograms(original_image, sha512_image):
    """
    Create a comparison plot of both histograms.
    """
    plt.figure(figsize=(15, 5))
    
    # Original image
    plt.subplot(1, 3, 1)
    plt.imshow(original_image, cmap='gray')
    plt.title('Original Lenna Image')
    plt.axis('off')
    
    # SHA512 image
    plt.subplot(1, 3, 2)
    plt.imshow(sha512_image, cmap='gray')
    plt.title('SHA512-Based Image')
    plt.axis('off')
    
    # Comparison of histograms
    plt.subplot(1, 3, 3)
    hist_original = cv2.calcHist([original_image], [0], None, [256], [0, 256])
    hist_sha512 = cv2.calcHist([sha512_image], [0], None, [256], [0, 256])
    
    plt.plot(hist_original, color='black', label='Original', alpha=0.7)
    plt.plot(hist_sha512, color='red', label='SHA512', alpha=0.7)
    plt.title('Histogram Comparison')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.xlim([0, 256])
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('comparison_histogram.png', dpi=300, bbox_inches='tight')
    print("Comparison histogram saved as 'comparison_histogram.png'")
    plt.show()

def main():
    """
    Main function to execute the program.
    """
    print("=" * 60)
    print("Image Histogram and SHA512 Analysis")
    print("=" * 60)
    
    # Load Lenna grayscale image
    print("\n1. Loading Lenna grayscale image...")
    lenna_gray = load_lenna_grayscale()
    print(f"   Image shape: {lenna_gray.shape}")
    print(f"   Image dtype: {lenna_gray.dtype}")
    
    # Plot original histogram
    print("\n2. Plotting original image histogram...")
    plot_original_histogram(lenna_gray)
    
    # Apply SHA512 and plot
    print("\n3. Applying SHA512 algorithm and plotting histogram...")
    sha512_image, hash_value = apply_sha512_and_plot(lenna_gray)
    
    # Compare both histograms
    print("\n4. Creating comparison plot...")
    compare_histograms(lenna_gray, sha512_image)
    
    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - original_histogram.png")
    print("  - sha512_histogram.png")
    print("  - comparison_histogram.png")
    print(f"\nSHA512 Hash: {hash_value}")

if __name__ == "__main__":
    main()
