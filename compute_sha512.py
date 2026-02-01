import hashlib
import os

def compute_sha512(file_path):
    """
    Compute SHA512 hash of a file.
    """
    sha512_hash = hashlib.sha512()
    
    # Read file in chunks to handle large files efficiently
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            sha512_hash.update(chunk)
    
    return sha512_hash.hexdigest()

if __name__ == "__main__":
    file_path = "lenna_padded.webp"
    
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found!")
        exit(1)
    
    file_size = os.path.getsize(file_path)
    sha512_hash = compute_sha512(file_path)
    
    print(f"File: {file_path}")
    print(f"File size: {file_size} bytes")
    print(f"SHA512: {sha512_hash}")
