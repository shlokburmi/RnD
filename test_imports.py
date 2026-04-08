print("Starting import test...")
try:
    import numpy as np
    print("✓ NumPy imported")
except Exception as e:
    print(f"✗ NumPy error: {e}")

try:
    import cv2
    print("✓ OpenCV imported")
except Exception as e:
    print(f"✗ OpenCV error: {e}")

try:
    import matplotlib.pyplot as plt
    print("✓ Matplotlib imported")
except Exception as e:
    print(f"✗ Matplotlib error: {e}")

try:
    from PIL import Image
    print("✓ PIL imported")
except Exception as e:
    print(f"✗ PIL error: {e}")

print("Import test complete!")
