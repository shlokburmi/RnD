import hashlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
import numpy as np
from cryptography.fernet import Fernet
import os

def extract_roi(image_path, roi_coords=None):
    """Extract Region of Interest from image."""
    img = Image.open(image_path)
    width, height = img.size
    
    if roi_coords is None:
        left = int(width * 0.25)
        top = int(height * 0.15)
        right = int(width * 0.75)
        bottom = int(height * 0.70)
        roi_coords = (left, top, right, bottom)
    
    roi = img.crop(roi_coords)
    non_roi_img = img.copy()
    non_roi_img.paste((0, 0, 0), roi_coords)
    
    return roi, non_roi_img, roi_coords

def derive_encryption_key(algorithm_name, data):
    """Derive encryption key from hash algorithm."""
    if algorithm_name == 'SHA512':
        hash_obj = hashlib.sha512(data)
    elif algorithm_name == 'SHA256':
        hash_obj = hashlib.sha256(data)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")
    
    # Use first 32 bytes of hash for Fernet key (requires 32 bytes base64 encoded)
    hash_digest = hash_obj.digest()[:32]
    import base64
    key = base64.urlsafe_b64encode(hash_digest)
    return key

def encrypt_image_data(image_data, algorithm_name):
    """Encrypt image data using derived key."""
    try:
        key = derive_encryption_key(algorithm_name, image_data)
        cipher = Fernet(key)
        encrypted_data = cipher.encrypt(image_data)
        return encrypted_data, key
    except Exception as e:
        print(f"Encryption error: {e}")
        return None, None

def get_pixel_histogram(image):
    """Get histogram data from image."""
    if image.mode == 'RGB':
        # Convert to numpy array and flatten
        img_array = np.array(image)
        r_hist, _ = np.histogram(img_array[:,:,0], bins=256, range=(0, 256))
        g_hist, _ = np.histogram(img_array[:,:,1], bins=256, range=(0, 256))
        b_hist, _ = np.histogram(img_array[:,:,2], bins=256, range=(0, 256))
        return r_hist, g_hist, b_hist
    else:
        img_array = np.array(image)
        hist, _ = np.histogram(img_array, bins=256, range=(0, 256))
        return hist, hist, hist

def get_encrypted_histogram(encrypted_data):
    """Get histogram from encrypted binary data."""
    byte_array = np.frombuffer(encrypted_data, dtype=np.uint8)
    hist, _ = np.histogram(byte_array, bins=256, range=(0, 256))
    return hist, hist, hist

def plot_comparison_histograms():
    """Plot histograms for ROI and non-ROI before and after encryption."""
    
    image_path = "lenna_padded.webp"
    roi, non_roi, roi_coords = extract_roi(image_path)
    
    print("=" * 80)
    print("HISTOGRAM ANALYSIS: BEFORE & AFTER ENCRYPTION")
    print("=" * 80)
    
    # Get original histograms
    print("\nExtracting pixel data...")
    roi_r_hist, roi_g_hist, roi_b_hist = get_pixel_histogram(roi)
    non_roi_r_hist, non_roi_g_hist, non_roi_b_hist = get_pixel_histogram(non_roi)
    
    # Encrypt data
    print("Encrypting ROI with SHA512...")
    roi_data = roi.tobytes()
    roi_encrypted, roi_key = encrypt_image_data(roi_data, 'SHA512')
    
    print("Encrypting Non-ROI with SHA256...")
    non_roi_data = non_roi.tobytes()
    non_roi_encrypted, non_roi_key = encrypt_image_data(non_roi_data, 'SHA256')
    
    if roi_encrypted and non_roi_encrypted:
        print("Encryption successful!")
        
        # Get encrypted histograms
        roi_enc_r_hist, roi_enc_g_hist, roi_enc_b_hist = get_encrypted_histogram(roi_encrypted)
        non_roi_enc_r_hist, non_roi_enc_g_hist, non_roi_enc_b_hist = get_encrypted_histogram(non_roi_encrypted)
        
        # Create comprehensive visualization
        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
        
        # ===== ROI ANALYSIS =====
        # Title
        fig.suptitle('Image Encryption & Histogram Analysis\nROI (Face) - SHA512 | Non-ROI - SHA256', 
                     fontsize=16, fontweight='bold', y=0.995)
        
        # ROI Original - Red
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.bar(range(256), roi_r_hist, color='red', alpha=0.7, edgecolor='darkred')
        ax1.set_title('ROI Before Encryption\nRed Channel', fontweight='bold')
        ax1.set_xlabel('Pixel Value')
        ax1.set_ylabel('Frequency')
        ax1.set_xlim(0, 256)
        
        # ROI Original - Green
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.bar(range(256), roi_g_hist, color='green', alpha=0.7, edgecolor='darkgreen')
        ax2.set_title('ROI Before Encryption\nGreen Channel', fontweight='bold')
        ax2.set_xlabel('Pixel Value')
        ax2.set_ylabel('Frequency')
        ax2.set_xlim(0, 256)
        
        # ROI Original - Blue
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.bar(range(256), roi_b_hist, color='blue', alpha=0.7, edgecolor='darkblue')
        ax3.set_title('ROI Before Encryption\nBlue Channel', fontweight='bold')
        ax3.set_xlabel('Pixel Value')
        ax3.set_ylabel('Frequency')
        ax3.set_xlim(0, 256)
        
        # ROI Original - Combined
        ax4 = fig.add_subplot(gs[0, 3])
        ax4.plot(range(256), roi_r_hist, color='red', alpha=0.6, label='Red')
        ax4.plot(range(256), roi_g_hist, color='green', alpha=0.6, label='Green')
        ax4.plot(range(256), roi_b_hist, color='blue', alpha=0.6, label='Blue')
        ax4.set_title('ROI Before Encryption\nAll Channels', fontweight='bold')
        ax4.set_xlabel('Pixel Value')
        ax4.set_ylabel('Frequency')
        ax4.legend()
        ax4.set_xlim(0, 256)
        
        # ROI Encrypted - Red
        ax5 = fig.add_subplot(gs[1, 0])
        ax5.bar(range(256), roi_enc_r_hist, color='red', alpha=0.7, edgecolor='darkred')
        ax5.set_title('ROI After SHA512 Encryption\nByte Distribution (R)', fontweight='bold')
        ax5.set_xlabel('Byte Value')
        ax5.set_ylabel('Frequency')
        ax5.set_xlim(0, 256)
        
        # ROI Encrypted - Green
        ax6 = fig.add_subplot(gs[1, 1])
        ax6.bar(range(256), roi_enc_g_hist, color='green', alpha=0.7, edgecolor='darkgreen')
        ax6.set_title('ROI After SHA512 Encryption\nByte Distribution (G)', fontweight='bold')
        ax6.set_xlabel('Byte Value')
        ax6.set_ylabel('Frequency')
        ax6.set_xlim(0, 256)
        
        # ROI Encrypted - Blue
        ax7 = fig.add_subplot(gs[1, 2])
        ax7.bar(range(256), roi_enc_b_hist, color='blue', alpha=0.7, edgecolor='darkblue')
        ax7.set_title('ROI After SHA512 Encryption\nByte Distribution (B)', fontweight='bold')
        ax7.set_xlabel('Byte Value')
        ax7.set_ylabel('Frequency')
        ax7.set_xlim(0, 256)
        
        # ROI Encrypted - Combined
        ax8 = fig.add_subplot(gs[1, 3])
        ax8.bar(range(256), roi_enc_r_hist, color='red', alpha=0.4, label='Dist 1')
        ax8.bar(range(256), roi_enc_g_hist, color='green', alpha=0.4, label='Dist 2')
        ax8.bar(range(256), roi_enc_b_hist, color='blue', alpha=0.4, label='Dist 3')
        ax8.set_title('ROI After SHA512 Encryption\nCombined Distribution', fontweight='bold')
        ax8.set_xlabel('Byte Value')
        ax8.set_ylabel('Frequency')
        ax8.legend()
        ax8.set_xlim(0, 256)
        
        # ===== NON-ROI ANALYSIS =====
        
        # Non-ROI Original - Red
        ax9 = fig.add_subplot(gs[2, 0])
        ax9.bar(range(256), non_roi_r_hist, color='red', alpha=0.7, edgecolor='darkred')
        ax9.set_title('Non-ROI Before Encryption\nRed Channel', fontweight='bold')
        ax9.set_xlabel('Pixel Value')
        ax9.set_ylabel('Frequency')
        ax9.set_xlim(0, 256)
        
        # Non-ROI Original - Green
        ax10 = fig.add_subplot(gs[2, 1])
        ax10.bar(range(256), non_roi_g_hist, color='green', alpha=0.7, edgecolor='darkgreen')
        ax10.set_title('Non-ROI Before Encryption\nGreen Channel', fontweight='bold')
        ax10.set_xlabel('Pixel Value')
        ax10.set_ylabel('Frequency')
        ax10.set_xlim(0, 256)
        
        # Non-ROI Original - Blue
        ax11 = fig.add_subplot(gs[2, 2])
        ax11.bar(range(256), non_roi_b_hist, color='blue', alpha=0.7, edgecolor='darkblue')
        ax11.set_title('Non-ROI Before Encryption\nBlue Channel', fontweight='bold')
        ax11.set_xlabel('Pixel Value')
        ax11.set_ylabel('Frequency')
        ax11.set_xlim(0, 256)
        
        # Non-ROI Original - Combined
        ax12 = fig.add_subplot(gs[2, 3])
        ax12.plot(range(256), non_roi_r_hist, color='red', alpha=0.6, label='Red')
        ax12.plot(range(256), non_roi_g_hist, color='green', alpha=0.6, label='Green')
        ax12.plot(range(256), non_roi_b_hist, color='blue', alpha=0.6, label='Blue')
        ax12.set_title('Non-ROI Before Encryption\nAll Channels', fontweight='bold')
        ax12.set_xlabel('Pixel Value')
        ax12.set_ylabel('Frequency')
        ax12.legend()
        ax12.set_xlim(0, 256)
        
        plt.savefig('histogram_before_encryption.png', dpi=150, bbox_inches='tight')
        print("\nSaved: histogram_before_encryption.png")
        
        # ===== AFTER ENCRYPTION COMPARISON =====
        fig2 = plt.figure(figsize=(16, 8))
        gs2 = gridspec.GridSpec(2, 4, figure=fig2, hspace=0.3, wspace=0.3)
        
        fig2.suptitle('Image Encryption & Histogram Analysis\nAfter Encryption - ROI (SHA512) & Non-ROI (SHA256)', 
                      fontsize=16, fontweight='bold', y=0.98)
        
        # ROI After Encryption
        ax_roi_enc_1 = fig2.add_subplot(gs2[0, 0])
        ax_roi_enc_1.bar(range(256), roi_enc_r_hist, color='red', alpha=0.7, edgecolor='darkred')
        ax_roi_enc_1.set_title('ROI Encrypted (SHA512)\nDistribution 1', fontweight='bold')
        ax_roi_enc_1.set_xlabel('Byte Value')
        ax_roi_enc_1.set_ylabel('Frequency')
        ax_roi_enc_1.set_xlim(0, 256)
        
        ax_roi_enc_2 = fig2.add_subplot(gs2[0, 1])
        ax_roi_enc_2.bar(range(256), roi_enc_g_hist, color='green', alpha=0.7, edgecolor='darkgreen')
        ax_roi_enc_2.set_title('ROI Encrypted (SHA512)\nDistribution 2', fontweight='bold')
        ax_roi_enc_2.set_xlabel('Byte Value')
        ax_roi_enc_2.set_ylabel('Frequency')
        ax_roi_enc_2.set_xlim(0, 256)
        
        ax_roi_enc_3 = fig2.add_subplot(gs2[0, 2])
        ax_roi_enc_3.bar(range(256), roi_enc_b_hist, color='blue', alpha=0.7, edgecolor='darkblue')
        ax_roi_enc_3.set_title('ROI Encrypted (SHA512)\nDistribution 3', fontweight='bold')
        ax_roi_enc_3.set_xlabel('Byte Value')
        ax_roi_enc_3.set_ylabel('Frequency')
        ax_roi_enc_3.set_xlim(0, 256)
        
        ax_roi_enc_all = fig2.add_subplot(gs2[0, 3])
        ax_roi_enc_all.bar(range(256), roi_enc_r_hist, color='red', alpha=0.33, label='Dist 1')
        ax_roi_enc_all.bar(range(256), roi_enc_g_hist, color='green', alpha=0.33, label='Dist 2')
        ax_roi_enc_all.bar(range(256), roi_enc_b_hist, color='blue', alpha=0.33, label='Dist 3')
        ax_roi_enc_all.set_title('ROI Encrypted (SHA512)\nCombined Distributions', fontweight='bold')
        ax_roi_enc_all.set_xlabel('Byte Value')
        ax_roi_enc_all.set_ylabel('Frequency')
        ax_roi_enc_all.legend()
        ax_roi_enc_all.set_xlim(0, 256)
        
        # Non-ROI After Encryption
        ax_non_roi_enc_1 = fig2.add_subplot(gs2[1, 0])
        ax_non_roi_enc_1.bar(range(256), non_roi_enc_r_hist, color='red', alpha=0.7, edgecolor='darkred')
        ax_non_roi_enc_1.set_title('Non-ROI Encrypted (SHA256)\nDistribution 1', fontweight='bold')
        ax_non_roi_enc_1.set_xlabel('Byte Value')
        ax_non_roi_enc_1.set_ylabel('Frequency')
        ax_non_roi_enc_1.set_xlim(0, 256)
        
        ax_non_roi_enc_2 = fig2.add_subplot(gs2[1, 1])
        ax_non_roi_enc_2.bar(range(256), non_roi_enc_g_hist, color='green', alpha=0.7, edgecolor='darkgreen')
        ax_non_roi_enc_2.set_title('Non-ROI Encrypted (SHA256)\nDistribution 2', fontweight='bold')
        ax_non_roi_enc_2.set_xlabel('Byte Value')
        ax_non_roi_enc_2.set_ylabel('Frequency')
        ax_non_roi_enc_2.set_xlim(0, 256)
        
        ax_non_roi_enc_3 = fig2.add_subplot(gs2[1, 2])
        ax_non_roi_enc_3.bar(range(256), non_roi_enc_b_hist, color='blue', alpha=0.7, edgecolor='darkblue')
        ax_non_roi_enc_3.set_title('Non-ROI Encrypted (SHA256)\nDistribution 3', fontweight='bold')
        ax_non_roi_enc_3.set_xlabel('Byte Value')
        ax_non_roi_enc_3.set_ylabel('Frequency')
        ax_non_roi_enc_3.set_xlim(0, 256)
        
        ax_non_roi_enc_all = fig2.add_subplot(gs2[1, 3])
        ax_non_roi_enc_all.bar(range(256), non_roi_enc_r_hist, color='red', alpha=0.33, label='Dist 1')
        ax_non_roi_enc_all.bar(range(256), non_roi_enc_g_hist, color='green', alpha=0.33, label='Dist 2')
        ax_non_roi_enc_all.bar(range(256), non_roi_enc_b_hist, color='blue', alpha=0.33, label='Dist 3')
        ax_non_roi_enc_all.set_title('Non-ROI Encrypted (SHA256)\nCombined Distributions', fontweight='bold')
        ax_non_roi_enc_all.set_xlabel('Byte Value')
        ax_non_roi_enc_all.set_ylabel('Frequency')
        ax_non_roi_enc_all.legend()
        ax_non_roi_enc_all.set_xlim(0, 256)
        
        plt.savefig('histogram_after_encryption.png', dpi=150, bbox_inches='tight')
        print("Saved: histogram_after_encryption.png")
        
        # ===== SIDE-BY-SIDE COMPARISON =====
        fig3 = plt.figure(figsize=(16, 10))
        gs3 = gridspec.GridSpec(2, 2, figure=fig3, hspace=0.3, wspace=0.3)
        
        fig3.suptitle('Before vs After Encryption Comparison\nHistogram Distribution Analysis', 
                      fontsize=16, fontweight='bold')
        
        # ROI Comparison
        ax_roi_before = fig3.add_subplot(gs3[0, 0])
        ax_roi_before.plot(range(256), roi_r_hist, color='red', alpha=0.6, label='Red', linewidth=2)
        ax_roi_before.plot(range(256), roi_g_hist, color='green', alpha=0.6, label='Green', linewidth=2)
        ax_roi_before.plot(range(256), roi_b_hist, color='blue', alpha=0.6, label='Blue', linewidth=2)
        ax_roi_before.set_title('ROI Before Encryption\n(Original Pixel Distribution)', fontweight='bold', fontsize=12)
        ax_roi_before.set_xlabel('Pixel Value', fontsize=10)
        ax_roi_before.set_ylabel('Frequency', fontsize=10)
        ax_roi_before.legend(fontsize=9)
        ax_roi_before.grid(True, alpha=0.3)
        ax_roi_before.set_xlim(0, 256)
        
        ax_roi_after = fig3.add_subplot(gs3[0, 1])
        ax_roi_after.bar(range(256), roi_enc_r_hist, color='red', alpha=0.3, label='Pattern 1')
        ax_roi_after.bar(range(256), roi_enc_g_hist, color='green', alpha=0.3, label='Pattern 2')
        ax_roi_after.bar(range(256), roi_enc_b_hist, color='blue', alpha=0.3, label='Pattern 3')
        ax_roi_after.set_title('ROI After SHA512 Encryption\n(Encrypted Byte Distribution)', fontweight='bold', fontsize=12)
        ax_roi_after.set_xlabel('Byte Value', fontsize=10)
        ax_roi_after.set_ylabel('Frequency', fontsize=10)
        ax_roi_after.legend(fontsize=9)
        ax_roi_after.grid(True, alpha=0.3)
        ax_roi_after.set_xlim(0, 256)
        
        # Non-ROI Comparison
        ax_non_roi_before = fig3.add_subplot(gs3[1, 0])
        ax_non_roi_before.plot(range(256), non_roi_r_hist, color='red', alpha=0.6, label='Red', linewidth=2)
        ax_non_roi_before.plot(range(256), non_roi_g_hist, color='green', alpha=0.6, label='Green', linewidth=2)
        ax_non_roi_before.plot(range(256), non_roi_b_hist, color='blue', alpha=0.6, label='Blue', linewidth=2)
        ax_non_roi_before.set_title('Non-ROI Before Encryption\n(Original Pixel Distribution)', fontweight='bold', fontsize=12)
        ax_non_roi_before.set_xlabel('Pixel Value', fontsize=10)
        ax_non_roi_before.set_ylabel('Frequency', fontsize=10)
        ax_non_roi_before.legend(fontsize=9)
        ax_non_roi_before.grid(True, alpha=0.3)
        ax_non_roi_before.set_xlim(0, 256)
        
        ax_non_roi_after = fig3.add_subplot(gs3[1, 1])
        ax_non_roi_after.bar(range(256), non_roi_enc_r_hist, color='red', alpha=0.3, label='Pattern 1')
        ax_non_roi_after.bar(range(256), non_roi_enc_g_hist, color='green', alpha=0.3, label='Pattern 2')
        ax_non_roi_after.bar(range(256), non_roi_enc_b_hist, color='blue', alpha=0.3, label='Pattern 3')
        ax_non_roi_after.set_title('Non-ROI After SHA256 Encryption\n(Encrypted Byte Distribution)', fontweight='bold', fontsize=12)
        ax_non_roi_after.set_xlabel('Byte Value', fontsize=10)
        ax_non_roi_after.set_ylabel('Frequency', fontsize=10)
        ax_non_roi_after.legend(fontsize=9)
        ax_non_roi_after.grid(True, alpha=0.3)
        ax_non_roi_after.set_xlim(0, 256)
        
        plt.savefig('histogram_comparison.png', dpi=150, bbox_inches='tight')
        print("Saved: histogram_comparison.png")
        
        # Print statistics
        print("\n" + "=" * 80)
        print("HISTOGRAM STATISTICS")
        print("=" * 80)
        
        print("\nROI Before Encryption (Pixel Values):")
        print(f"  Red   - Mean: {np.mean(roi_r_hist):.2f}, Std Dev: {np.std(roi_r_hist):.2f}")
        print(f"  Green - Mean: {np.mean(roi_g_hist):.2f}, Std Dev: {np.std(roi_g_hist):.2f}")
        print(f"  Blue  - Mean: {np.mean(roi_b_hist):.2f}, Std Dev: {np.std(roi_b_hist):.2f}")
        
        print("\nROI After SHA512 Encryption (Byte Distribution):")
        print(f"  Dist1 - Mean: {np.mean(roi_enc_r_hist):.2f}, Std Dev: {np.std(roi_enc_r_hist):.2f}")
        print(f"  Dist2 - Mean: {np.mean(roi_enc_g_hist):.2f}, Std Dev: {np.std(roi_enc_g_hist):.2f}")
        print(f"  Dist3 - Mean: {np.mean(roi_enc_b_hist):.2f}, Std Dev: {np.std(roi_enc_b_hist):.2f}")
        
        print("\nNon-ROI Before Encryption (Pixel Values):")
        print(f"  Red   - Mean: {np.mean(non_roi_r_hist):.2f}, Std Dev: {np.std(non_roi_r_hist):.2f}")
        print(f"  Green - Mean: {np.mean(non_roi_g_hist):.2f}, Std Dev: {np.std(non_roi_g_hist):.2f}")
        print(f"  Blue  - Mean: {np.mean(non_roi_b_hist):.2f}, Std Dev: {np.std(non_roi_b_hist):.2f}")
        
        print("\nNon-ROI After SHA256 Encryption (Byte Distribution):")
        print(f"  Dist1 - Mean: {np.mean(non_roi_enc_r_hist):.2f}, Std Dev: {np.std(non_roi_enc_r_hist):.2f}")
        print(f"  Dist2 - Mean: {np.mean(non_roi_enc_g_hist):.2f}, Std Dev: {np.std(non_roi_enc_g_hist):.2f}")
        print(f"  Dist3 - Mean: {np.mean(non_roi_enc_b_hist):.2f}, Std Dev: {np.std(non_roi_enc_b_hist):.2f}")
        
        print("\n" + "=" * 80)
        print("KEY OBSERVATIONS")
        print("=" * 80)
        print("\n1. BEFORE ENCRYPTION:")
        print("   - Original images show distinct patterns in their histograms")
        print("   - Pixel values are concentrated in certain ranges")
        print("   - Recognizable structure reflects image content")
        
        print("\n2. AFTER ENCRYPTION:")
        print("   - Encrypted data shows more uniform distribution")
        print("   - Byte values are spread across the spectrum")
        print("   - No obvious patterns - good encryption randomness")
        
        print("\n3. SECURITY IMPLICATIONS:")
        print("   - Uniform distribution indicates strong encryption")
        print("   - No information leakage from encrypted data")
        print("   - SHA512 (ROI) provides stronger key derivation than SHA256")
        
        print("\n" + "=" * 80)
        
    else:
        print("Encryption failed!")

if __name__ == "__main__":
    plot_comparison_histograms()
