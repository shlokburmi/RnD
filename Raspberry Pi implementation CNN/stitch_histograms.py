import os
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def generate_report_histogram():
    dir_plain = r"s:\NIIT\Sem 6\R&D\RAND\RnD\Raspberry Pi implementation CNN\histograms_plain"
    dir_enc = r"s:\NIIT\Sem 6\R&D\RAND\RnD\Raspberry Pi implementation CNN\histograms_encrypted"
    
    # Get arbitrary first image from the plain directory
    plain_files = glob.glob(os.path.join(dir_plain, "*_hist.png"))
    if not plain_files:
        print("No plain histogram images found.")
        return
        
    plain_img_path = plain_files[0]
    # The corresponding encrypted histogram has `_hist_enc.png` instead of `_hist.png`
    base_name = os.path.basename(plain_img_path).replace("_hist.png", "")
    enc_img_path = os.path.join(dir_enc, f"{base_name}_hist_enc.png")
    
    if not os.path.exists(enc_img_path):
        # Fallback to the first available in enc dir
        enc_files = glob.glob(os.path.join(dir_enc, "*.png"))
        if not enc_files:
            print("No encrypted histogram images found.")
            return
        enc_img_path = enc_files[0]
        
    img_plain = mpimg.imread(plain_img_path)
    img_enc = mpimg.imread(enc_img_path)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # We display them with the extent so that the axes show intuitive values.
    # The OpenCV canvases are 400x512.
    # X-axis will represent intensity [0, 255].
    # Y-axis represents normalized frequency (0 = bottom, 1.0 = top max).
    
    axes[0].imshow(img_plain, extent=[0, 255, 0, 1.0], aspect='auto')
    axes[0].set_title(f"A. Plain Image Histogram", fontsize=14, pad=10)
    axes[0].set_xlabel("Pixel Intensity", fontsize=12)
    axes[0].set_ylabel("Normalized Frequency", fontsize=12)
    
    axes[1].imshow(img_enc, extent=[0, 255, 0, 1.0], aspect='auto')
    axes[1].set_title(f"B. Encrypted Image Histogram", fontsize=14, pad=10)
    axes[1].set_xlabel("Pixel Intensity", fontsize=12)
    axes[1].set_ylabel("Normalized Frequency", fontsize=12)
    
    plt.suptitle("Comparative Analysis of Histograms Before & After Encryption (Sample from 6000 Cases)", fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    output_path = r"s:\NIIT\Sem 6\R&D\RAND\RnD\Raspberry Pi implementation CNN\final_histogram_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Generated successfully: {output_path}")

if __name__ == '__main__':
    generate_report_histogram()
