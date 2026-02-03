import cv2
import numpy as np

def calculate_avalanche_effect(img1_path, img2_path):
    # Load images as grayscale
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    
    if img1 is None or img2 is None:
        print(f"Error loading images: {img1_path}, {img2_path}")
        return

    # Flatten to 1D arrays
    arr1 = img1.ravel()
    arr2 = img2.ravel()
    
    # Calculate Bit Change Rate (Avalanche Effect)
    # Convert numpy array to bytes, then to bits
    # Efficient method: XOR the arrays, then count set bits in the result
    
    # Bitwise XOR of the pixel values
    xor_diff = np.bitwise_xor(arr1, arr2)
    
    # Count set bits (population count) for each byte
    # We can use a lookup table or built-in functionality.
    # Python's bin().count('1') is simple but slow for large arrays.
    # np.unpackbits is better.
    
    bits_diff = np.unpackbits(xor_diff.astype(np.uint8))
    total_bits = len(bits_diff)
    diff_count = np.sum(bits_diff)
    
    avalanche_effect = (diff_count / total_bits) * 100
    
    print(f"--- Avalanche Effect Analysis ---")
    print(f"Image 1: {img1_path}")
    print(f"Image 2: {img2_path}")
    print(f"Total Bits: {total_bits}")
    print(f"Different Bits: {diff_count}")
    print(f"Avalanche Effect: {avalanche_effect:.4f}%")
    print(f"Ideal Value: ~50.00%")

if __name__ == "__main__":
    calculate_avalanche_effect("lenna_encrypted_k1.png", "lenna_encrypted_k2.png")
