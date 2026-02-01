import hashlib
import time
from PIL import Image
import os

def extract_roi(image_path, roi_coords=None):
    """
    Extract Region of Interest (ROI) from image.
    Default ROI is the face area (approximate center portion of the image).
    roi_coords: tuple of (left, top, right, bottom) in pixels
    """
    img = Image.open(image_path)
    width, height = img.size
    
    # Default ROI: face region (center portion, approximate coordinates)
    # For Lenna 512x512, face is roughly in the center-upper region
    if roi_coords is None:
        left = int(width * 0.25)      # 25% from left
        top = int(height * 0.15)       # 15% from top
        right = int(width * 0.75)      # 75% from left
        bottom = int(height * 0.70)    # 70% from top
        roi_coords = (left, top, right, bottom)
    
    roi = img.crop(roi_coords)
    non_roi_img = img.copy()
    
    # Create mask for non-ROI (rest of the image)
    mask = Image.new('L', img.size, 255)
    mask.paste(0, roi_coords)
    
    return roi, non_roi_img, mask, roi_coords

def compute_hash(data, algorithm='sha512'):
    """
    Compute hash of data using specified algorithm.
    """
    if algorithm.lower() == 'sha512':
        hasher = hashlib.sha512()
    elif algorithm.lower() == 'sha256':
        hasher = hashlib.sha256()
    elif algorithm.lower() == 'md5':
        hasher = hashlib.md5()
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    if isinstance(data, str):
        data = open(data, 'rb').read()
    elif isinstance(data, Image.Image):
        data = data.tobytes()
    
    hasher.update(data)
    return hasher.hexdigest()

def get_hash_output_size(algorithm):
    """Return output size in bits for hash algorithm."""
    sizes = {
        'sha512': 512,
        'sha256': 256,
        'md5': 128
    }
    return sizes.get(algorithm.lower(), 0)

def compare_hashes():
    """
    Extract ROI from padded image, apply SHA512 to ROI,
    apply SHA256 to non-ROI, and compare algorithms.
    """
    image_path = "lenna_padded.webp"
    
    print("=" * 80)
    print("REGION OF INTEREST (ROI) ANALYSIS - LENNA PADDED IMAGE")
    print("=" * 80)
    
    # Load image
    img = Image.open(image_path)
    print(f"\nImage: {image_path}")
    print(f"Image size: {img.size[0]}x{img.size[1]} pixels")
    print(f"Image mode: {img.mode}")
    
    # Extract ROI and non-ROI
    roi, full_img, mask, roi_coords = extract_roi(image_path)
    roi_left, roi_top, roi_right, roi_bottom = roi_coords
    roi_width = roi_right - roi_left
    roi_height = roi_bottom - roi_top
    roi_area = roi_width * roi_height
    full_area = img.size[0] * img.size[1]
    non_roi_area = full_area - roi_area
    
    print(f"\nROI (Face) Coordinates: ({roi_left}, {roi_top}, {roi_right}, {roi_bottom})")
    print(f"ROI dimensions: {roi_width}x{roi_height} pixels")
    print(f"ROI area: {roi_area} pixels ({roi_area*100/full_area:.2f}% of image)")
    print(f"Non-ROI area: {non_roi_area} pixels ({non_roi_area*100/full_area:.2f}% of image)")
    
    # Save ROI and non-ROI for inspection
    roi.save("roi_face.webp")
    print(f"\nROI saved as: roi_face.webp")
    
    # Create non-ROI image
    non_roi_img = img.copy()
    non_roi_img.paste((0, 0, 0), roi_coords)  # Fill ROI with black
    non_roi_img.save("non_roi.webp")
    print(f"Non-ROI saved as: non_roi.webp")
    
    # Get data sizes
    roi_data = roi.tobytes()
    non_roi_data = non_roi_img.tobytes()
    
    roi_size = len(roi_data)
    non_roi_size = len(non_roi_data)
    
    print(f"\nROI data size: {roi_size} bytes")
    print(f"Non-ROI data size: {non_roi_size} bytes")
    
    # ========== SHA512 on ROI ==========
    print("\n" + "=" * 80)
    print("SHA512 HASHING - ROI (Face)")
    print("=" * 80)
    
    start_time = time.time()
    roi_hash_sha512 = compute_hash(roi_data, 'sha512')
    roi_sha512_time = time.time() - start_time
    
    print(f"SHA512 Hash (ROI): {roi_hash_sha512}")
    print(f"Execution time: {roi_sha512_time*1000:.4f} ms")
    print(f"Output size: {get_hash_output_size('sha512')} bits ({get_hash_output_size('sha512')//8} bytes)")
    print(f"Hash rate: {roi_size / (roi_sha512_time * 1024 * 1024):.2f} MB/s")
    
    # ========== SHA256 on Non-ROI ==========
    print("\n" + "=" * 80)
    print("SHA256 HASHING - Non-ROI (Rest of Image)")
    print("=" * 80)
    
    start_time = time.time()
    non_roi_hash_sha256 = compute_hash(non_roi_data, 'sha256')
    non_roi_sha256_time = time.time() - start_time
    
    print(f"SHA256 Hash (Non-ROI): {non_roi_hash_sha256}")
    print(f"Execution time: {non_roi_sha256_time*1000:.4f} ms")
    print(f"Output size: {get_hash_output_size('sha256')} bits ({get_hash_output_size('sha256')//8} bytes)")
    print(f"Hash rate: {non_roi_size / (non_roi_sha256_time * 1024 * 1024):.2f} MB/s")
    
    # ========== Additional hashes for comparison ==========
    print("\n" + "=" * 80)
    print("ADDITIONAL HASH ALGORITHMS (For Comparison)")
    print("=" * 80)
    
    # MD5 on ROI
    print("\nMD5 (ROI):")
    start_time = time.time()
    roi_hash_md5 = compute_hash(roi_data, 'md5')
    roi_md5_time = time.time() - start_time
    print(f"Hash: {roi_hash_md5}")
    print(f"Time: {roi_md5_time*1000:.4f} ms")
    print(f"Output: {get_hash_output_size('md5')} bits")
    
    # SHA256 on ROI (for comparison with SHA512)
    print("\nSHA256 (ROI):")
    start_time = time.time()
    roi_hash_sha256 = compute_hash(roi_data, 'sha256')
    roi_sha256_time = time.time() - start_time
    print(f"Hash: {roi_hash_sha256}")
    print(f"Time: {roi_sha256_time*1000:.4f} ms")
    print(f"Output: {get_hash_output_size('sha256')} bits")
    
    # ========== Comparison Table ==========
    print("\n" + "=" * 80)
    print("COMPREHENSIVE COMPARISON - PARAMETER ANALYSIS")
    print("=" * 80)
    
    algorithms_data = [
        {
            'name': 'SHA512',
            'region': 'ROI (Face)',
            'hash': roi_hash_sha512,
            'time': roi_sha512_time,
            'output_bits': 512,
            'security': 'Very High',
            'collision_resistant': 'Yes',
            'data_size': roi_size
        },
        {
            'name': 'SHA256',
            'region': 'Non-ROI',
            'hash': non_roi_hash_sha256,
            'time': non_roi_sha256_time,
            'output_bits': 256,
            'security': 'High',
            'collision_resistant': 'Yes',
            'data_size': non_roi_size
        },
        {
            'name': 'SHA256',
            'region': 'ROI (Face)',
            'hash': roi_hash_sha256,
            'time': roi_sha256_time,
            'output_bits': 256,
            'security': 'High',
            'collision_resistant': 'Yes',
            'data_size': roi_size
        },
        {
            'name': 'MD5',
            'region': 'ROI (Face)',
            'hash': roi_hash_md5,
            'time': roi_md5_time,
            'output_bits': 128,
            'security': 'Low (Broken)',
            'collision_resistant': 'No',
            'data_size': roi_size
        }
    ]
    
    print(f"\n{'Algorithm':<10} {'Region':<15} {'Time (ms)':<12} {'Output (bits)':<14} {'Security':<20} {'Speed (MB/s)':<12}")
    print("-" * 90)
    
    for algo in algorithms_data:
        speed_mbs = algo['data_size'] / (algo['time'] * 1024 * 1024) if algo['time'] > 0 else 0
        print(f"{algo['name']:<10} {algo['region']:<15} {algo['time']*1000:<12.4f} {algo['output_bits']:<14} {algo['security']:<20} {speed_mbs:<12.2f}")
    
    # ========== Time Complexity Analysis ==========
    print("\n" + "=" * 80)
    print("TIME COMPLEXITY & PERFORMANCE ANALYSIS")
    print("=" * 80)
    
    print("\nSHA512 vs SHA256 (same ROI data):")
    ratio = roi_sha512_time / roi_sha256_time if roi_sha256_time > 0 else 0
    print(f"  SHA512 time: {roi_sha512_time*1000:.4f} ms")
    print(f"  SHA256 time: {roi_sha256_time*1000:.4f} ms")
    print(f"  Ratio (SHA512/SHA256): {ratio:.2f}x")
    print(f"  Time complexity: O(n) for both (linear with input data)")
    
    print("\nSpace Complexity:")
    print(f"  SHA512: O(1) - constant memory (fixed state)")
    print(f"  SHA256: O(1) - constant memory (fixed state)")
    print(f"  MD5:    O(1) - constant memory (fixed state)")
    
    # ========== Security Analysis ==========
    print("\n" + "=" * 80)
    print("SECURITY ANALYSIS")
    print("=" * 80)
    
    print("\nAlgorithm Security Levels:")
    print("  1. SHA512 (512-bit): ✓ RECOMMENDED")
    print("     - Security strength: 256 bits against collisions")
    print("     - Resistance: Strong against pre-image attacks")
    print("     - Status: Secure and widely trusted")
    print()
    print("  2. SHA256 (256-bit): ✓ RECOMMENDED")
    print("     - Security strength: 128 bits against collisions")
    print("     - Resistance: Strong against pre-image attacks")
    print("     - Status: Part of SHA-2 family, widely used")
    print()
    print("  3. MD5 (128-bit): ✗ NOT RECOMMENDED")
    print("     - Security strength: Broken, collisions found")
    print("     - Resistance: Weak, vulnerable to collision attacks")
    print("     - Status: Deprecated, should not be used for security")
    
    # ========== Summary ==========
    print("\n" + "=" * 80)
    print("SUMMARY & RECOMMENDATIONS")
    print("=" * 80)
    
    total_time = roi_sha512_time + non_roi_sha256_time
    print(f"\nCombined Processing Time:")
    print(f"  SHA512 (ROI):      {roi_sha512_time*1000:.4f} ms")
    print(f"  SHA256 (Non-ROI):  {non_roi_sha256_time*1000:.4f} ms")
    print(f"  Total:             {total_time*1000:.4f} ms")
    
    print(f"\nHash Combination Results:")
    print(f"  ROI Hash (SHA512):     {roi_hash_sha512}")
    print(f"  Non-ROI Hash (SHA256): {non_roi_hash_sha256}")
    
    print(f"\nRecommendations:")
    print(f"  ✓ Use SHA512 for ROI (Face) - Maximum security for critical region")
    print(f"  ✓ Use SHA256 for Non-ROI - Good balance of speed and security")
    print(f"  ✗ Avoid MD5 - Cryptographically broken, only for non-security purposes")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    compare_hashes()
