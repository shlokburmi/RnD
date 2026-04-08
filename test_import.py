print("Starting test...")
try:
    import cv2
    print("cv2 imported successfully")
except Exception as e:
    print(f"cv2 import failed: {e}")

try:
    import numpy as np
    print("numpy imported successfully")
except Exception as e:
    print(f"numpy import  failed: {e}")

try:
    import hashlib
    print("hashlib imported successfully")
except Exception as e:
    print(f"hashlib import failed: {e}")

print("All imports tested!")
