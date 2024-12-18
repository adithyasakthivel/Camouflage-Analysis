import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to compute histogram for an image
def compute_histogram(image_path):
    # Read the image
    image = cv2.imread(image_path)
    
    # Convert the image from BGR to RGB for matplotlib display
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Split the image into its three channels: R, G, B
    channels = cv2.split(image)
    
    # Parameters for the histogram
    hist_size = [256]  # Number of bins
    hist_range = [0, 256]  # Range of pixel values
    
    # Calculate histograms for each color channel
    hist_r = cv2.calcHist([channels[2]], [0], None, hist_size, hist_range)  # Red channel
    hist_g = cv2.calcHist([channels[1]], [0], None, hist_size, hist_range)  # Green channel
    hist_b = cv2.calcHist([channels[0]], [0], None, hist_size, hist_range)  # Blue channel
    
    return hist_r, hist_g, hist_b, image_rgb

# Function to compare histograms using correlation method
def compare_histograms(hist1, hist2):
    # Use the correlation method to compare histograms
    score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return score

# Function to plot the histograms of two images side by side
def plot_histograms(hist1, hist2, title1, title2):
    fig, ax = plt.subplots(2, 3, figsize=(12, 6))

    # Plot histograms for first image
    ax[0, 0].plot(hist1[0], color='r')
    ax[0, 0].set_title(f"{title1} - Red Channel")
    
    ax[0, 1].plot(hist1[1], color='g')
    ax[0, 1].set_title(f"{title1} - Green Channel")
    
    ax[0, 2].plot(hist1[2], color='b')
    ax[0, 2].set_title(f"{title1} - Blue Channel")

    # Plot histograms for second image
    ax[1, 0].plot(hist2[0], color='r')
    ax[1, 0].set_title(f"{title2} - Red Channel")
    
    ax[1, 1].plot(hist2[1], color='g')
    ax[1, 1].set_title(f"{title2} - Green Channel")
    
    ax[1, 2].plot(hist2[2], color='b')
    ax[1, 2].set_title(f"{title2} - Blue Channel")

    plt.tight_layout()
    plt.show()

# Paths to your images
image1_path = r"Camouflage.jpg"
image2_path = r"Location.jpg"

# Compute histograms for both images
hist_r1, hist_g1, hist_b1, image_rgb1 = compute_histogram(image1_path)
hist_r2, hist_g2, hist_b2, image_rgb2 = compute_histogram(image2_path)

# Combine histograms into a single array for comparison
hist1_combined = [hist_r1, hist_g1, hist_b1]
hist2_combined = [hist_r2, hist_g2, hist_b2]

# Compare histograms using correlation
r_score = compare_histograms(hist_r1, hist_r2)
g_score = compare_histograms(hist_g1, hist_g2)
b_score = compare_histograms(hist_b1, hist_b2)

# Display similarity scores
print(f"Red channel similarity: {r_score}")
print(f"Green channel similarity: {g_score}")
print(f"Blue channel similarity: {b_score}")

# Average similarity score
A_score = (r_score + g_score + b_score)/3
print(f"Average RGB similarity: {A_score:.2f}")

# Plot histograms of both images for comparison
plot_histograms(hist1_combined, hist2_combined, 'Image 1', 'Image 2')
