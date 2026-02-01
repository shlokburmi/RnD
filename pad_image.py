import hashlib
from PIL import Image
import os

def pad_image_for_sha512(input_path, output_path):
    """
    Pad an image to be compatible with SHA512 block size (1024-bit = 128 bytes).
    
    SHA512 operates on 1024-bit (128-byte) blocks. This function ensures
    the image data is a multiple of 128 bytes.
    """
    
    # Open the image
    img = Image.open(input_path)
    print(f"Original image size: {img.size}")
    print(f"Original image mode: {img.mode}")
    
    # Get image dimensions
    width, height = img.size
    
    # Calculate new dimensions to be multiples of 128 bytes
    # For simplicity, we'll calculate based on bytes needed per row
    # assuming 3 bytes per pixel (RGB)
    
    # Calculate total pixels needed as multiple of 128
    total_pixels = width * height
    bytes_per_pixel = 3 if img.mode == 'RGB' else (4 if img.mode == 'RGBA' else 1)
    total_bytes = total_pixels * bytes_per_pixel
    
    # Round up to nearest multiple of 128
    padded_bytes = ((total_bytes + 127) // 128) * 128
    padded_pixels = padded_bytes // bytes_per_pixel
    
    # Calculate new dimensions (maintain aspect ratio approximately)
    new_width = width
    new_height = (padded_pixels + width - 1) // width
    
    print(f"\nOriginal bytes: {total_bytes}")
    print(f"Padded bytes: {padded_bytes}")
    print(f"Original dimensions: {width}x{height}")
    print(f"New dimensions: {new_width}x{new_height}")
    
    # Create new image with padding
    if img.mode == 'RGBA':
        padded_img = Image.new('RGBA', (new_width, new_height), (0, 0, 0, 0))
    else:
        padded_img = Image.new(img.mode, (new_width, new_height), 0)
    
    # Paste original image
    padded_img.paste(img, (0, 0))
    
    # Save padded image
    padded_img.save(output_path, 'webp')
    print(f"\nPadded image saved to: {output_path}")
    
    # Calculate SHA512 hash of original and padded image data
    with open(input_path, 'rb') as f:
        original_hash = hashlib.sha512(f.read()).hexdigest()
    
    with open(output_path, 'rb') as f:
        padded_hash = hashlib.sha512(f.read()).hexdigest()
    
    print(f"\nOriginal image SHA512: {original_hash}")
    print(f"Padded image SHA512: {padded_hash}")
    print(f"\nOriginal file size: {os.path.getsize(input_path)} bytes")
    print(f"Padded file size: {os.path.getsize(output_path)} bytes")
    
    return padded_img

if __name__ == "__main__":
    input_file = "lenna.webp"
    output_file = "lenna_padded.webp"
    
    pad_image_for_sha512(input_file, output_file)
