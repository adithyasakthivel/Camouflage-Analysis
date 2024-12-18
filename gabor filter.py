import cv2
import numpy as np

def compare_gabor_filters(img1, img2, num_scales=4, num_orientations=6):
  """Compares the Gabor filters of two images.

  Args:
    img1: The first image.
    img2: The second image.
    num_scales: The number of scales to use for the Gabor filter.
    num_orientations: The number of orientations to use for the Gabor filter.

  Returns:
    A list of tuples containing the scale and orientation for each Gabor filter, along with the
    correlation coefficient between the filtered images, and the average correlation.
  """

  # Convert images to grayscale
  gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
  gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

  # Normalize images
  gray1 = cv2.normalize(gray1, None, 0, 255, cv2.NORM_MINMAX)
  gray2 = cv2.normalize(gray2, None, 0, 255, cv2.NORM_MINMAX)

  # Create Gabor filter kernels
  kernels = []
  for theta in np.arange(0, np.pi, np.pi / num_orientations):
    for sigma in np.arange(0.5, 4.0, 2.0 / num_scales):
      kernel = cv2.getGaborKernel((21, 21), sigma, theta, 10.0, 0.5, 0)
      kernels.append(kernel)

  # Apply Gabor filters to images
  filtered1 = [cv2.filter2D(gray1, -1, kernel) for kernel in kernels]
  filtered2 = [cv2.filter2D(gray2, -1, kernel) for kernel in kernels]

  # Calculate correlation coefficients
  results = []
  correlation_values = []
  for i in range(len(kernels)):
    corr = cv2.matchTemplate(filtered1[i], filtered2[i], cv2.TM_CCOEFF_NORMED)
    results.append((i // num_orientations, i % num_orientations, corr.max()))
    correlation_values.append(corr.max())

  # Calculate average correlation
  average_correlation = np.mean(correlation_values)

  return results, average_correlation

# Example usage
img1 = cv2.imread(r"Camouflage.jpg")
img2 = cv2.imread(r"Location.jpg")
results, average_correlation = compare_gabor_filters(img1, img2)

# Print results
for scale, orientation, corr in results:
  print(f"Scale: {scale}, Orientation: {orientation}, Correlation: {corr:.2f}")

print(f"Average Correlation: {average_correlation:.2f}")
