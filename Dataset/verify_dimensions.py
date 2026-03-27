import os
import cv2

processed_folder = "Processed Dataset"
output_file = "convertedresults.txt"

total_images = 0
correct_size_images = 0
failed_images = []

with open(output_file, 'w', encoding='utf-8') as log:
    log.write("=" * 60 + "\n")
    log.write("   PROCESSED DATASET DIMENSION VERIFICATION REPORT\n")
    log.write("=" * 60 + "\n\n")

    if not os.path.exists(processed_folder):
        log.write(f"ERROR: The folder '{processed_folder}' does not exist.\n")
    else:
        # Loop over each subfolder (MRI, CTScan, etc.)
        for category in os.listdir(processed_folder):
            category_path = os.path.join(processed_folder, category)
            if not os.path.isdir(category_path):
                continue
                
            log.write(f"\n[{category.upper()}]\n")
            log.write("-" * 60 + "\n")
            
            for file in os.listdir(category_path):
                if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                    
                total_images += 1
                img_path = os.path.join(category_path, file)
                
                # Check dimensions using OpenCV
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    log.write(f"❌ {file:<25} (UNREADABLE)\n")
                    failed_images.append(f"{category}/{file}")
                    continue
                    
                h, w = img.shape
                
                # Verify exactly 256x256
                if h == 256 and w == 256:
                    correct_size_images += 1
                    log.write(f"✅ {file:<25} | {w}x{h}\n")
                else:
                    log.write(f"❌ {file:<25} | {w}x{h} (FAIL - Needs 256x256)\n")
                    failed_images.append(f"{category}/{file} [{w}x{h}]")

    # Add Final Summary Block 
    log.write("\n" + "=" * 60 + "\n")
    log.write("                      SUMMARY\n")
    log.write("=" * 60 + "\n")
    log.write(f"Total Images Checked   : {total_images}\n")
    log.write(f"Correctly 256x256      : {correct_size_images}\n")
    
    success_rate = (correct_size_images / total_images * 100) if total_images > 0 else 0
    log.write(f"Success Rate           : {success_rate:.2f}%\n")
    
    if len(failed_images) > 0:
        log.write("\nWARNING - Anomalies Found:\n")
        for fail in failed_images:
            log.write(f"  - {fail}\n")
    else:
        log.write("\nSUCCESS: All processed images perfectly conform to the 256x256 grayscale benchmark.\n")

print(f"Dimension verification complete! Report saved to '{output_file}' in the Dataset folder.")
