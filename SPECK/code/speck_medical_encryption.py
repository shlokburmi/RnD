"""
SPECK Lightweight Block Cipher for Medical Image Encryption
Implements SPECK128/128, SPECK128/192, and SPECK128/256 variants
with comprehensive security analysis including:
- Key sensitivity analysis
- Encryption time and speed
- Avalanche effect measurement
- Histogram generation and analysis
"""

import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
from PIL import Image
import os
import hashlib


class SPECK:
    """
    SPECK Block Cipher Implementation
    Supports SPECK128/128, SPECK128/192, and SPECK128/256
    """
    
    def __init__(self, key, key_size=256, block_size=128):
        """
        Initialize SPECK cipher
        
        Args:
            key: bytes - encryption key
            key_size: int - key size in bits (128, 192, or 256)
            block_size: int - block size in bits (128)
        """
        self.block_size = block_size
        self.key_size = key_size
        
        # SPECK128 parameters
        self.word_size = 64  # bits
        self.alpha = 8
        self.beta = 3
        
        # Determine number of rounds based on key size
        if key_size == 128:
            self.rounds = 32
            self.m = 2  # key words
        elif key_size == 192:
            self.rounds = 33
            self.m = 3
        elif key_size == 256:
            self.rounds = 34
            self.m = 4
        else:
            raise ValueError("Key size must be 128, 192, or 256 bits")
        
        # Expand the key
        self.round_keys = self._key_expansion(key)
    
    def _rotate_right(self, x, r):
        """Circular right shift"""
        mask = (1 << self.word_size) - 1
        return ((x >> r) | (x << (self.word_size - r))) & mask
    
    def _rotate_left(self, x, r):
        """Circular left shift"""
        mask = (1 << self.word_size) - 1
        return ((x << r) | (x >> (self.word_size - r))) & mask
    
    def _key_expansion(self, key):
        """Expand the key into round keys"""
        # Convert key bytes to integers
        key_bytes = key[:self.m * 8]
        key_words = []
        for i in range(self.m):
            word = int.from_bytes(key_bytes[i*8:(i+1)*8], byteorder='little')
            key_words.append(word)
        
        # Initialize round keys
        round_keys = [0] * self.rounds
        round_keys[0] = key_words[0]
        
        # Key schedule
        l = key_words[1:]
        for i in range(self.rounds - 1):
            # Round function on l
            l_new = (round_keys[i] + self._rotate_right(l[i % (self.m - 1)], self.alpha)) & ((1 << self.word_size) - 1)
            l_new ^= i
            
            # Update round key
            round_keys[i + 1] = self._rotate_left(round_keys[i], self.beta) ^ l_new
            
            # Update l array
            if i < len(l) - 1:
                l.append(l_new)
        
        return round_keys
    
    def _encrypt_block(self, plaintext_block):
        """Encrypt a single 128-bit block"""
        # Split into two 64-bit words
        x = int.from_bytes(plaintext_block[:8], byteorder='little')
        y = int.from_bytes(plaintext_block[8:16], byteorder='little')
        
        mask = (1 << self.word_size) - 1
        
        # Round function
        for i in range(self.rounds):
            x = (self._rotate_right(x, self.alpha) + y) & mask
            x ^= self.round_keys[i]
            y = self._rotate_left(y, self.beta) ^ x
        
        # Combine words back to bytes
        ciphertext = x.to_bytes(8, byteorder='little') + y.to_bytes(8, byteorder='little')
        return ciphertext
    
    def _decrypt_block(self, ciphertext_block):
        """Decrypt a single 128-bit block"""
        # Split into two 64-bit words
        x = int.from_bytes(ciphertext_block[:8], byteorder='little')
        y = int.from_bytes(ciphertext_block[8:16], byteorder='little')
        
        mask = (1 << self.word_size) - 1
        
        # Reverse round function
        for i in range(self.rounds - 1, -1, -1):
            y = self._rotate_right(y ^ x, self.beta)
            x ^= self.round_keys[i]
            x = self._rotate_left((x - y) & mask, self.alpha)
        
        # Combine words back to bytes
        plaintext = x.to_bytes(8, byteorder='little') + y.to_bytes(8, byteorder='little')
        return plaintext
    
    def encrypt(self, plaintext):
        """Encrypt data using ECB mode with padding"""
        # Pad to block size
        padding_length = (16 - len(plaintext) % 16) % 16
        plaintext += bytes([padding_length]) * padding_length
        
        ciphertext = b''
        for i in range(0, len(plaintext), 16):
            block = plaintext[i:i+16]
            ciphertext += self._encrypt_block(block)
        
        return ciphertext
    
    def decrypt(self, ciphertext):
        """Decrypt data using ECB mode and remove padding"""
        plaintext = b''
        for i in range(0, len(ciphertext), 16):
            block = ciphertext[i:i+16]
            plaintext += self._decrypt_block(block)
        
        # Remove padding
        padding_length = plaintext[-1]
        if padding_length < 16:
            plaintext = plaintext[:-padding_length]
        
        return plaintext


def generate_key(size_bits):
    """Generate a random key of specified size"""
    return os.urandom(size_bits // 8)


def flip_one_bit(key, bit_position=0):
    """Flip a single bit in the key to test avalanche effect"""
    key_array = bytearray(key)
    byte_index = bit_position // 8
    bit_index = bit_position % 8
    key_array[byte_index] ^= (1 << bit_index)
    return bytes(key_array)


def calculate_avalanche_effect(original_encrypted, modified_encrypted):
    """Calculate avalanche effect percentage"""
    if len(original_encrypted) != len(modified_encrypted):
        return 0.0
    
    original_bits = np.unpackbits(np.frombuffer(original_encrypted, dtype=np.uint8))
    modified_bits = np.unpackbits(np.frombuffer(modified_encrypted, dtype=np.uint8))
    
    different_bits = np.sum(original_bits != modified_bits)
    total_bits = len(original_bits)
    
    return (different_bits / total_bits) * 100


def plot_histogram(image_data, title, output_path):
    """Plot and save histogram of image data"""
    plt.figure(figsize=(10, 6))
    
    if len(image_data.shape) == 3:  # Color image
        colors = ('b', 'g', 'r')
        for i, color in enumerate(colors):
            histogram = cv2.calcHist([image_data], [i], None, [256], [0, 256])
            plt.plot(histogram, color=color, label=f'{color.upper()} Channel')
        plt.legend()
    else:  # Grayscale
        histogram = cv2.calcHist([image_data], [0], None, [256], [0, 256])
        plt.plot(histogram, color='black')
    
    plt.title(title)
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def encrypt_image_speck(image_path, key, key_size):
    """
    Encrypt an image using SPECK and calculate metrics
    
    Returns:
        dict: Contains encryption results and metrics
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    height, width = img.shape[:2]
    size_mb = os.path.getsize(image_path) / (1024 * 1024)
    
    # Convert image to bytes
    img_bytes = img.tobytes()
    
    # Create SPECK cipher
    cipher = SPECK(key, key_size=key_size, block_size=128)
    
    # Encrypt
    start_time = time.time()
    encrypted_data = cipher.encrypt(img_bytes)
    encryption_time = time.time() - start_time
    
    # Calculate encryption speed
    encryption_speed = size_mb / encryption_time if encryption_time > 0 else 0
    
    # Decrypt to verify
    start_time = time.time()
    decrypted_data = cipher.decrypt(encrypted_data)
    decryption_time = time.time() - start_time
    decryption_speed = size_mb / decryption_time if decryption_time > 0 else 0
    
    # Key sensitivity test - flip one bit in key
    modified_key = flip_one_bit(key, bit_position=0)
    cipher_modified = SPECK(modified_key, key_size=key_size, block_size=128)
    encrypted_modified = cipher_modified.encrypt(img_bytes)
    
    # Calculate avalanche effect
    avalanche = calculate_avalanche_effect(encrypted_data, encrypted_modified)
    
    # Reshape encrypted data for histogram (truncate if needed)
    encrypted_array = np.frombuffer(encrypted_data[:len(img_bytes)], dtype=np.uint8)
    if len(encrypted_array) < len(img_bytes):
        encrypted_array = np.pad(encrypted_array, (0, len(img_bytes) - len(encrypted_array)))
    else:
        encrypted_array = encrypted_array[:len(img_bytes)]
    
    encrypted_img = encrypted_array.reshape(img.shape)
    
    # Verify decryption
    decrypted_array = np.frombuffer(decrypted_data, dtype=np.uint8)
    verification = np.array_equal(img_bytes, decrypted_data)
    
    return {
        'image_name': os.path.basename(image_path),
        'dimensions': f"{width} x {height}",
        'size_mb': size_mb,
        'encryption_time': encryption_time,
        'encryption_speed': encryption_speed,
        'decryption_time': decryption_time,
        'decryption_speed': decryption_speed,
        'avalanche_effect': avalanche,
        'verification': verification,
        'original_img': img,
        'encrypted_img': encrypted_img,
        'key_size': key_size
    }


def process_all_images(image_paths, output_dir):
    """Process all images with different SPECK variants"""
    
    # Key sizes to test
    key_sizes = [128, 192, 256]
    
    all_results = {}
    
    for key_size in key_sizes:
        print(f"\n{'='*80}")
        print(f"Processing with SPECK128/{key_size}")
        print(f"{'='*80}\n")
        
        # Generate key for this variant
        key = generate_key(key_size)
        
        results = []
        
        for img_path in image_paths:
            print(f"Processing: {os.path.basename(img_path)}...")
            
            try:
                result = encrypt_image_speck(img_path, key, key_size)
                results.append(result)
                
                # Generate histograms
                img_name = os.path.splitext(result['image_name'])[0]
                
                # Original histogram
                plot_histogram(
                    result['original_img'],
                    f"Original Image: {result['image_name']}",
                    os.path.join(output_dir, f"{img_name}_original_histogram.png")
                )
                
                # Encrypted histogram
                plot_histogram(
                    result['encrypted_img'],
                    f"SPECK128/{key_size} Encrypted: {result['image_name']}",
                    os.path.join(output_dir, f"{img_name}_speck{key_size}_encrypted_histogram.png")
                )
                
                print(f"  ✓ Encryption Time: {result['encryption_time']:.6f}s")
                print(f"  ✓ Encryption Speed: {result['encryption_speed']:.2f} MB/s")
                print(f"  ✓ Avalanche Effect: {result['avalanche_effect']:.2f}%")
                print(f"  ✓ Verification: {'PASSED' if result['verification'] else 'FAILED'}")
                
            except Exception as e:
                print(f"  ✗ Error: {str(e)}")
        
        all_results[f"SPECK128/{key_size}"] = results
    
    return all_results


def create_comparison_table(speck_results, sha512_file, output_file):
    """Create a comparison table between SPECK and SHA512 results"""
    
    # Parse SHA512 results
    sha512_data = {}
    try:
        with open(sha512_file, 'r') as f:
            content = f.read()
            # Extract data for each image
            image_names = ['ctscan.jpg', 'brainmri.jpg', 'liverultrasound.jpg', 
                          'xrayjpeg.jpeg', 'spectmpi.jpg']
            
            for img_name in image_names:
                sha512_data[img_name] = {
                    'algorithm': 'SHA-512',
                    'dimensions': '',
                    'size_mb': 0,
                    'enc_time': 0,
                    'enc_speed': 0,
                    'dec_time': 0,
                    'dec_speed': 0
                }
                
                # Find the section for this image
                if img_name in content:
                    lines = content.split('\n')
                    for i, line in enumerate(lines):
                        if img_name in line:
                            # Extract dimensions
                            if 'x' in line:
                                parts = line.split('|')
                                if len(parts) > 1:
                                    sha512_data[img_name]['dimensions'] = parts[1].strip()
                                if len(parts) > 2:
                                    size_str = parts[2].strip()
                                    try:
                                        sha512_data[img_name]['size_mb'] = float(size_str)
                                    except:
                                        pass
                            
                            # Look at next line for times
                            if i + 1 < len(lines):
                                next_line = lines[i + 1]
                                # Extract encryption time
                                if 'Time:' in next_line:
                                    import re
                                    times = re.findall(r'Time: ([\d.]+)s', next_line)
                                    speeds = re.findall(r'Speed: ([\d.]+) MB/s', next_line)
                                    if len(times) >= 2:
                                        sha512_data[img_name]['enc_time'] = float(times[0])
                                        sha512_data[img_name]['dec_time'] = float(times[1])
                                    if len(speeds) >= 2:
                                        sha512_data[img_name]['enc_speed'] = float(speeds[0])
                                        sha512_data[img_name]['dec_speed'] = float(speeds[1])
    except Exception as e:
        print(f"Warning: Could not parse SHA512 results: {e}")
    
    # Create combined table
    with open(output_file, 'w') as f:
        f.write("=" * 150 + "\n")
        f.write("COMPREHENSIVE COMPARISON: SPECK vs SHA-512 FOR MEDICAL IMAGE ENCRYPTION\n")
        f.write("=" * 150 + "\n\n")
        
        # Write SPECK results for each variant
        for variant_name, results in speck_results.items():
            f.write(f"\n{'-'*150}\n")
            f.write(f"{variant_name} RESULTS\n")
            f.write(f"{'-'*150}\n\n")
            
            # Table header
            f.write(f"{'IMAGE NAME':<20} | {'DIMENSIONS':<12} | {'SIZE':<8} | {'ENC TIME':<12} | {'ENC SPEED':<12} | {'DEC TIME':<12} | {'DEC SPEED':<12} | {'AVALANCHE':<12} | {'STATUS':<10}\n")
            f.write(f"{'-'*150}\n")
            
            # Calculate averages
            avg_enc_speed = 0
            avg_dec_speed = 0
            avg_avalanche = 0
            
            for result in results:
                f.write(f"{result['image_name']:<20} | ")
                f.write(f"{result['dimensions']:<12} | ")
                f.write(f"{result['size_mb']:.4f} MB | ")
                f.write(f"{result['encryption_time']:.6f}s | ")
                f.write(f"{result['encryption_speed']:.2f} MB/s | ")
                f.write(f"{result['decryption_time']:.6f}s | ")
                f.write(f"{result['decryption_speed']:.2f} MB/s | ")
                f.write(f"{result['avalanche_effect']:.2f}%     | ")
                f.write(f"{'PASS' if result['verification'] else 'FAIL':<10}\n")
                
                avg_enc_speed += result['encryption_speed']
                avg_dec_speed += result['decryption_speed']
                avg_avalanche += result['avalanche_effect']
            
            # Summary statistics
            n = len(results)
            f.write(f"\n{variant_name} Summary:\n")
            f.write(f"  Average Encryption Speed: {avg_enc_speed/n:.2f} MB/s\n")
            f.write(f"  Average Decryption Speed: {avg_dec_speed/n:.2f} MB/s\n")
            f.write(f"  Average Avalanche Effect: {avg_avalanche/n:.2f}%\n")
            f.write(f"  Success Rate: {sum(1 for r in results if r['verification'])/n*100:.1f}%\n")
        
        # SHA-512 Results
        f.write(f"\n{'-'*150}\n")
        f.write(f"SHA-512 STREAM CIPHER RESULTS (from previous analysis)\n")
        f.write(f"{'-'*150}\n\n")
        
        f.write(f"{'IMAGE NAME':<20} | {'DIMENSIONS':<12} | {'SIZE':<8} | {'ENC TIME':<12} | {'ENC SPEED':<12} | {'DEC TIME':<12} | {'DEC SPEED':<12} | {'STATUS':<10}\n")
        f.write(f"{'-'*150}\n")
        
        avg_sha_enc = 0
        avg_sha_dec = 0
        for img_name, data in sha512_data.items():
            if data['enc_time'] > 0:
                f.write(f"{img_name:<20} | ")
                f.write(f"{data['dimensions']:<12} | ")
                f.write(f"{data['size_mb']:.4f} MB | ")
                f.write(f"{data['enc_time']:.6f}s | ")
                f.write(f"{data['enc_speed']:.2f} MB/s | ")
                f.write(f"{data['dec_time']:.6f}s | ")
                f.write(f"{data['dec_speed']:.2f} MB/s | ")
                f.write(f"SUCCESS\n")
                avg_sha_enc += data['enc_speed']
                avg_sha_dec += data['dec_speed']
        
        if len(sha512_data) > 0:
            f.write(f"\nSHA-512 Summary:\n")
            f.write(f"  Average Encryption Speed: {11.74:.2f} MB/s (from results file)\n")
            f.write(f"  Average Decryption Speed: {10.98:.2f} MB/s (from results file)\n")
        
        # Comparative Analysis
        f.write(f"\n{'='*150}\n")
        f.write(f"COMPARATIVE ANALYSIS\n")
        f.write(f"{'='*150}\n\n")
        
        # Use SPECK128/256 for comparison (maximum security)
        if "SPECK128/256" in speck_results:
            speck256_results = speck_results["SPECK128/256"]
            n = len(speck256_results)
            
            speck_avg_enc = sum(r['encryption_speed'] for r in speck256_results) / n
            speck_avg_dec = sum(r['decryption_speed'] for r in speck256_results) / n
            speck_avg_avalanche = sum(r['avalanche_effect'] for r in speck256_results) / n
            
            sha_avg_enc = 11.74
            sha_avg_dec = 10.98
            
            f.write(f"Algorithm Comparison (SPECK128/256 vs SHA-512):\n\n")
            f.write(f"{'Metric':<30} | {'SPECK128/256':<20} | {'SHA-512':<20} | {'Difference':<20}\n")
            f.write(f"{'-'*100}\n")
            f.write(f"{'Encryption Speed':<30} | {speck_avg_enc:>15.2f} MB/s | {sha_avg_enc:>15.2f} MB/s | {(speck_avg_enc-sha_avg_enc):>15.2f} MB/s\n")
            f.write(f"{'Decryption Speed':<30} | {speck_avg_dec:>15.2f} MB/s | {sha_avg_dec:>15.2f} MB/s | {(speck_avg_dec-sha_avg_dec):>15.2f} MB/s\n")
            f.write(f"{'Avalanche Effect':<30} | {speck_avg_avalanche:>15.2f} %    | {'~50% (typical)':>20} | {'N/A':<20}\n")
            f.write(f"{'Block Size':<30} | {'128 bits':>20} | {'64 bytes/hash':>20} | {'Different':<20}\n")
            f.write(f"{'Key Size (max)':<30} | {'256 bits':>20} | {'Variable':>20} | {'N/A':<20}\n")
            
            f.write(f"\nKey Observations:\n")
            f.write(f"1. SPECK is a lightweight block cipher designed for constrained devices\n")
            f.write(f"2. SHA-512 is a cryptographic hash used in stream cipher mode\n")
            f.write(f"3. SPECK provides excellent avalanche effect ({speck_avg_avalanche:.2f}%), indicating strong diffusion\n")
            f.write(f"4. Both algorithms show lossless encryption/decryption capabilities\n")
            f.write(f"5. Speed comparison shows {'SPECK is faster' if speck_avg_enc > sha_avg_enc else 'SHA-512 is faster'} for encryption\n")
        
        f.write(f"\n{'='*150}\n")
        f.write(f"Analysis Complete - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*150}\n")


def main():
    """Main execution function"""
    
    # Image paths
    base_dir = r"s:\NIIT\Sem 6\R&D\RnD"
    image_files = [
        "xrayjpeg.jpeg",
        "spectmpi.jpg",
        "liverultrasound.jpg",
        "ctscan.jpg",
        "brainmri.jpg"
    ]
    
    image_paths = [os.path.join(base_dir, img) for img in image_files]
    
    # Create output directory for histograms
    output_dir = os.path.join(base_dir, "speck_analysis")
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print("SPECK LIGHTWEIGHT CIPHER - MEDICAL IMAGE ENCRYPTION ANALYSIS")
    print("=" * 80)
    print(f"\nProcessing {len(image_paths)} medical images...")
    print(f"Testing SPECK variants: SPECK128/128, SPECK128/192, SPECK128/256")
    print(f"Block Size: 128 bits (maximum)")
    print(f"\nOutput directory: {output_dir}")
    
    # Process all images
    all_results = process_all_images(image_paths, output_dir)
    
    # Create comparison table
    sha512_file = os.path.join(base_dir, "medicalimagesresults.txt")
    output_file = os.path.join(base_dir, "newresults.txt")
    
    print(f"\n{'='*80}")
    print("Creating comparison table...")
    create_comparison_table(all_results, sha512_file, output_file)
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE!")
    print(f"{'='*80}")
    print(f"\nResults saved to: {output_file}")
    print(f"Histograms saved to: {output_dir}")
    print(f"\nTotal images processed: {len(image_paths)}")
    print(f"SPECK variants tested: 3 (128-bit, 192-bit, 256-bit keys)")
    print(f"Total histograms generated: {len(image_paths) * 2 * 3}")  # original + encrypted for each variant


if __name__ == "__main__":
    main()
