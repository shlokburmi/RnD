import cv2
import numpy as np
import matplotlib.pyplot as plt

def plot_image_histogram(image_path):
    """
    Load an image and plot its histogram.
    
    Args:
        image_path: Path to the image file
    """
    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    print(f"Image loaded successfully!")
    print(f"Image shape: {image.shape}")
    print(f"Image dtype: {image.dtype}")
    print(f"Min pixel value: {image.min()}")
    print(f"Max pixel value: {image.max()}")
    print(f"Mean pixel value: {image.mean():.2f}")
    
    # Create figure with subplots
    plt.figure(figsize=(14, 6))
    
    # Display the grayscale image
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Lenna Grayscale Image', fontsize=14, fontweight='bold')
    plt.axis('off')
    
    # Calculate and plot histogram
    plt.subplot(1, 2, 2)
    histogram = cv2.calcHist([image], [0], None, [256], [0, 256])
    
    plt.plot(histogram, color='black', linewidth=2)
    plt.fill_between(range(256), histogram.flatten(), alpha=0.3, color='gray')
    plt.title('Histogram of Lenna Grayscale Image', fontsize=14, fontweight='bold')
    plt.xlabel('Pixel Intensity', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.xlim([0, 256])
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Add statistics text
    stats_text = f'Mean: {image.mean():.2f}\nStd: {image.std():.2f}\nMin: {image.min()}\nMax: {image.max()}'
    plt.text(0.98, 0.97, stats_text, 
             transform=plt.gca().transAxes,
             fontsize=10,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save the plot
    output_filename = 'lenna_grayscale_histogram.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"\nHistogram plot saved as '{output_filename}'")
    
    # Show the plot
    plt.show()
    
    # Also create a detailed histogram analysis
    print("\n" + "="*60)
    print("HISTOGRAM ANALYSIS")
    print("="*60)
    
    # Find peaks in histogram
    hist_array = histogram.flatten()
    peaks = []
    for i in range(1, len(hist_array)-1):
        if hist_array[i] > hist_array[i-1] and hist_array[i] > hist_array[i+1]:
            if hist_array[i] > 100:  # Only significant peaks
                peaks.append((i, hist_array[i]))
    
    peaks.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\nTop 5 Peak Intensities:")
    for idx, (intensity, count) in enumerate(peaks[:5], 1):
        print(f"  {idx}. Intensity {intensity}: {int(count)} pixels")
    
    # Analyze brightness distribution
    dark_pixels = np.sum(image < 85)
    mid_pixels = np.sum((image >= 85) & (image < 170))
    bright_pixels = np.sum(image >= 170)
    total_pixels = image.size
    
    print(f"\nBrightness Distribution:")
    print(f"  Dark pixels (0-84):     {dark_pixels:7d} ({dark_pixels/total_pixels*100:5.2f}%)")
    print(f"  Mid-tone pixels (85-169): {mid_pixels:7d} ({mid_pixels/total_pixels*100:5.2f}%)")
    print(f"  Bright pixels (170-255): {bright_pixels:7d} ({bright_pixels/total_pixels*100:5.2f}%)")
    print("="*60)

if __name__ == "__main__":
    # Path to the lenna grayscale image
    image_path = 'lenna_grayscale.webp'
    
    print("="*60)
    print("LENNA GRAYSCALE IMAGE HISTOGRAM ANALYSIS")
    print("="*60)
    print()
    
    plot_image_histogram(image_path)
