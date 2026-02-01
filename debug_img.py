from PIL import Image
import numpy as np
import os

path = "lenna.webp"
if not os.path.exists(path):
    print("File not found")
    exit(1)

print(f"Opening {path}")
img = Image.open(path).convert('RGB')
print(f"Mode: {img.mode}")
print(f"Size: {img.size}")

arr = np.array(img).copy()
print(f"Shape: {arr.shape}")
print(f"Dtype: {arr.dtype}")

try:
    val = arr[0, 0, 0]
    print(f"Value at [0,0,0]: {val}")
    
    new_val = (int(val) + 1) % 256
    print(f"New Value: {new_val}")
    
    arr[0, 0, 0] = new_val
    print("Assignment successful")
except Exception as e:
    print(f"Error: {e}")
