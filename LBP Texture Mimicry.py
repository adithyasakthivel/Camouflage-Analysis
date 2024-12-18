from PIL import Image, ImageOps
import numpy as np

# Define the function to calculate LBP manually
def calculate_lbp(image, radius=1, n_points=8):
    height, width = image.shape
    lbp_image = np.zeros((height, width), dtype=np.uint8)

    for i in range(radius, height - radius):
        for j in range(radius, width - radius):
            center = image[i, j]
            binary_string = ''
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    if dx != 0 or dy != 0:
                        binary_string += '1' if image[i + dy, j + dx] >= center else '0'
            lbp_image[i, j] = int(binary_string, 2)
    
    return lbp_image

# Load images using Pillow
img1 = Image.open(r"Camouflage.jpg")
img2 = Image.open(r"Location.jpg")

# Check if the images are loaded correctly
if img1 is None or img2 is None:
    print("Error: One or both images could not be loaded.")
else:
    print("Images loaded successfully.")

# Convert images to grayscale
gray1 = ImageOps.grayscale(img1)
gray2 = ImageOps.grayscale(img2)

# Convert to NumPy arrays
gray1 = np.array(gray1)
gray2 = np.array(gray2)

# Check if grayscale conversion worked
print(f"Gray1 shape: {gray1.shape}, Gray2 shape: {gray2.shape}")

# Calculate LBP features manually
lbp1 = calculate_lbp(gray1)
lbp2 = calculate_lbp(gray2)

print("LBP calculations completed")

# Compute histograms for LBP
hist1, _ = np.histogram(lbp1, bins=np.arange(0, 256), range=(0, 255))
hist2, _ = np.histogram(lbp2, bins=np.arange(0, 256), range=(0, 255))

# Normalize the histograms
hist1 = hist1.astype(float) / np.sum(hist1)
hist2 = hist2.astype(float) / np.sum(hist2)

# Compute histogram intersection
intersection = np.minimum(hist1, hist2)
similarity = np.sum(intersection)

# Print the final result
print("Texture similarity:", similarity)
