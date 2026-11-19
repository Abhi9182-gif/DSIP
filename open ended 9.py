import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the original image
image = cv2.imread(
  r"C:\Users\admin\Documents\DSIP\Lab Experiment\ALLU ARJUN.jpg"
)

# Check whether image is loaded
if image is None:
    raise FileNotFoundError("Image not found. Check the image path.")

# Convert BGR to RGB for Matplotlib
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# --------------------------------------------------
# 1. MEAN FILTER
# --------------------------------------------------

mean_filter = cv2.blur(image, (5, 5))
mean_rgb = cv2.cvtColor(mean_filter, cv2.COLOR_BGR2RGB)


# --------------------------------------------------
# 2. GAUSSIAN FILTER
# --------------------------------------------------

gaussian_filter = cv2.GaussianBlur(
    image,
    (5, 5),
    0
)

gaussian_rgb = cv2.cvtColor(
    gaussian_filter,
    cv2.COLOR_BGR2RGB
)


# --------------------------------------------------
# 3. MEDIAN FILTER
# --------------------------------------------------

median_filter = cv2.medianBlur(
    image,
    5
)

median_rgb = cv2.cvtColor(
    median_filter,
    cv2.COLOR_BGR2RGB
)


# --------------------------------------------------
# 4. LAPLACIAN SHARPENING
# --------------------------------------------------

# Convert image to grayscale only for calculating Laplacian
gray = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2GRAY
)

laplacian = cv2.Laplacian(
    gray,
    cv2.CV_64F
)

laplacian = np.uint8(
    np.absolute(laplacian)
)

# Convert Laplacian to 3 channels
laplacian_color = cv2.cvtColor(
    laplacian,
    cv2.COLOR_GRAY2BGR
)

# Add Laplacian details to original COLOR image
sharpened_laplacian = cv2.addWeighted(
    image,
    1.0,
    laplacian_color,
    1.0,
    0
)

laplacian_rgb = cv2.cvtColor(
    sharpened_laplacian,
    cv2.COLOR_BGR2RGB
)


# --------------------------------------------------
# 5. UNSHARP MASKING
# --------------------------------------------------

blur = cv2.GaussianBlur(
    image,
    (5, 5),
    0
)

unsharp = cv2.addWeighted(
    image,
    1.5,
    blur,
    -0.5,
    0
)

unsharp_rgb = cv2.cvtColor(
    unsharp,
    cv2.COLOR_BGR2RGB
)


# --------------------------------------------------
# DISPLAY ALL RESULTS
# --------------------------------------------------

plt.figure(figsize=(15, 10))

# Original
plt.subplot(2, 3, 1)
plt.imshow(image_rgb)
plt.title("Original")
plt.axis("off")

# Mean Filter
plt.subplot(2, 3, 2)
plt.imshow(mean_rgb)
plt.title("Mean Filter")
plt.axis("off")

# Gaussian Filter
plt.subplot(2, 3, 3)
plt.imshow(gaussian_rgb)
plt.title("Gaussian Filter")
plt.axis("off")

# Median Filter
plt.subplot(2, 3, 4)
plt.imshow(median_rgb)
plt.title("Median Filter")
plt.axis("off")

# Laplacian Sharpening
plt.subplot(2, 3, 5)
plt.imshow(laplacian_rgb)
plt.title("Laplacian Sharpening")
plt.axis("off")

# Unsharp Masking
plt.subplot(2, 3, 6)
plt.imshow(unsharp_rgb)
plt.title("Unsharp Masking")
plt.axis("off")

plt.tight_layout()
plt.show()