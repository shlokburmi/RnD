import cv2
import numpy as np
import hashlib
import matplotlib.pyplot as plt

def sha512_stream_encrypt(image, key_bytes):
    """
    Encrypts image using SHA-512 stream cipher logic.
    """
    height, width = image.shape
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
    keystream = keystream.reshape((height, width))
    
    return cv2.bitwise_xor(image, keystream)

def main():
    print("Generating Key Sensitivity Figure (Paper Methodology)...")
    
    # 1. Load Original Image (a)
    input_path = "lenna_grayscale.webp"
    image_a = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if image_a is None:
        print(f"Error: Could not load {input_path}")
        return

    # 2. Encrypt with Key 1 -> Image (b)
    key1 = b"SecretKey_For_Lenna_123"
    print(f"Encrypting with Key 1: {key1}")
    image_b = sha512_stream_encrypt(image_a, key1)
    
    # 3. Encrypt with Key 2 (1 bit modified) -> Image (c)
    key2 = b"SecretKey_For_Lenna_122" # '3' (00110011) -> '2' (00110010) is 1 bit flip
    print(f"Encrypting with Key 2: {key2}")
    image_c = sha512_stream_encrypt(image_a, key2)
    
    # 4. Generate Difference Image -> Image (d)
    # diff = |b - c|
    image_d = cv2.absdiff(image_b, image_c)
    
    # Calculate stats for the user to see (console only)
    diff_pixels = np.count_nonzero(image_d)
    total_pixels = image_d.size
    print(f"Difference Statistics:")
    print(f"Pixels Changed: {diff_pixels}/{total_pixels} ({(diff_pixels/total_pixels)*100:.2f}%)")
    
    # 5. Create the Figure matching the paper
    plt.figure(figsize=(16, 5))
    
    # (a) Original
    plt.subplot(1, 4, 1)
    plt.imshow(image_a, cmap='gray')
    plt.title('(a) Original Image')
    plt.axis('off')
    
    # (b) Encrypted (Key 1)
    plt.subplot(1, 4, 2)
    plt.imshow(image_b, cmap='gray')
    plt.title('(b) Encrypted (Key) ')
    plt.axis('off')
    
    # (c) Encrypted (Key + 1 bit)
    plt.subplot(1, 4, 3)
    plt.imshow(image_c, cmap='gray')
    plt.title('(c) Encrypted (Key changed)')
    plt.axis('off')
    
    # (d) Difference
    plt.subplot(1, 4, 4)
    plt.imshow(image_d, cmap='gray')
    plt.title('(d) Difference Image')
    plt.axis('off')
    
    output_file = 'key_sensitivity_comparison.png'
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"\nFigure saved as: {output_file}")
    
    # Also save the individual images for the record
    cv2.imwrite('lenna_encrypted_k1.png', image_b)
    cv2.imwrite('lenna_encrypted_k2.png', image_c)
    cv2.imwrite('lenna_diff_k1_k2.png', image_d)
    print("Individual images saved: lenna_encrypted_k1.png, lenna_encrypted_k2.png, lenna_diff_k1_k2.png")

    plt.show()

if __name__ == "__main__":
    main()
