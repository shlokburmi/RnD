import cv2
import os
import numpy as np

input_folder = "Normal Dataset"
output_folder = "Processed Dataset"

os.makedirs(output_folder, exist_ok=True)

def preprocess(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    # Resize properly (NO distortion)
    h, w = img.shape
    scale = 256 / max(h, w)

    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Padding
    canvas = np.zeros((256, 256), dtype=np.uint8)
    y_offset = (256 - new_h) // 2
    x_offset = (256 - new_w) // 2

    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

    return canvas

# Loop through all subdirectories
for category in os.listdir(input_folder):
    category_path = os.path.join(input_folder, category)
    if not os.path.isdir(category_path):
        continue
    
    # Create corresponding output subdirectory
    out_cat_path = os.path.join(output_folder, category)
    os.makedirs(out_cat_path, exist_ok=True)
    
    for file in os.listdir(category_path):
        if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
            
        path = os.path.join(category_path, file)
        processed_img = preprocess(path)
        
        if processed_img is not None:
            # Save as PNG as requested by the pipeline
            file_name = os.path.splitext(file)[0] + ".png"
            cv2.imwrite(os.path.join(out_cat_path, file_name), processed_img)
            print(f"Processed: {category}/{file_name}")