import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from speck_cnn_hybrid import IntegratedSecureSpeck

def generate_roi_report():
    img_path = r"s:\NIIT\Sem 6\R&D\RAND\RnD\BCSS_512\train_512\TCGA-A1-A0SK-DX1_xmin45749_ymin25055_MPP-0_0_512_size512.png"
    output_dir = r"s:\NIIT\Sem 6\R&D\RAND\RnD\Raspberry Pi implementation CNN"
    key = b"SecureEngine2026"
    
    print(f"Loading image from: {img_path}")
    raw = cv2.imread(img_path)
    if raw is None:
        print("Failed to load image. Check path.")
        return
        
    engine = IntegratedSecureSpeck(key)
    
    print("Encrypting with CNN-SPECK (Adaptive ROI extraction)...")
    enc_img, mask, enc_time = engine.encrypt_adaptive(raw)
    
    rgb_raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
    rgb_enc = cv2.cvtColor(enc_img, cv2.COLOR_BGR2RGB) if len(enc_img.shape) == 3 else cv2.cvtColor(enc_img, cv2.COLOR_GRAY2RGB)
    
    # Save individual images for the user
    cv2.imwrite(os.path.join(output_dir, "report_original.png"), raw)
    cv2.imwrite(os.path.join(output_dir, "report_roi_mask.png"), mask)
    cv2.imwrite(os.path.join(output_dir, "report_encrypted_cnn_speck.png"), enc_img)
    
    # Generate combined plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(rgb_raw)
    axes[0].set_title("1. Original Medical Image")
    axes[0].axis('off')
    
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title("2. CNN Extracted ROI Mask")
    axes[1].axis('off')
    
    axes[2].imshow(rgb_enc)
    axes[2].set_title(f"3. Encrypted Image (Time: {enc_time:.4f}s)")
    axes[2].axis('off')
    
    plt.suptitle("CNN-SPECK Hybrid Adaptive Encryption (Focus on ROI)", fontsize=16, fontweight='bold', y=1.05)
    plt.tight_layout()
    
    combined_path = os.path.join(output_dir, "report_combined_roi_encryption.png")
    plt.savefig(combined_path, dpi=300, bbox_inches='tight')
    print(f"Successfully generated comparison plot at: {combined_path}")

if __name__ == '__main__':
    generate_roi_report()
