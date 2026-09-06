import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.exposure import match_histograms

# 1. Folder Setup and Filenames
folder = r"C:\Users\admin\Documents\DSIP\Lab Experiment"

# Change to match the exact names of your 5 images:
image_filenames = [
    "pollen_washed.jpg",  # Image 1: Washed-out pollen
    "radial_spokes.png",  # Image 2: Radial circular pattern
    "pollen_dark.jpg",  # Image 3: Dark/underexposed pollen
    "dark_car.png",  # Image 4: Dark car with lamp
    "ct_scan.png",  # Image 5: Abdominal CT scan
]

image_paths = [os.path.join(folder, f) for f in image_filenames]


# 2. Helper Functions
def compute_he(img):
  """Computes Global Histogram Equalization."""
  return cv2.equalizeHist(img)


def create_gaussian_reference(
    shape, mean=128, std=45, low_val=0, high_val=255
):
  """Creates a synthetic image with a smooth Gaussian distribution."""
  np.random.seed(42)
  synth = np.random.normal(loc=mean, scale=std, size=shape)
  synth = np.clip(synth, low_val, high_val).astype(np.uint8)
  return synth


def compute_hm(source_img, reference_img):
  """Performs exact discrete histogram matching."""
  return match_histograms(source_img, reference_img)


# 3. Processing Pipeline
results = []

for idx, path in enumerate(image_paths, 1):
  img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
  if img is None:
    print(f"Skipping Image {idx}: File not found -> {path}")
    continue

  # Method A: Global Histogram Equalization
  he_img = compute_he(img)

  # Method B: Histogram Matching with Tailored Target References
  if idx == 1:
    # Image 1: Broad Gaussian mid-tone spread (mean=128, std=50)
    ref = create_gaussian_reference(img.shape, mean=128, std=50)
  elif idx == 2:
    # Image 2: Symmetric bell curve centered to balance spokes
    ref = create_gaussian_reference(img.shape, mean=120, std=40)
  elif idx == 3:
    # Image 3: Slightly right-shifted distribution to pull shadows up gently
    ref = create_gaussian_reference(img.shape, mean=135, std=45)
  elif idx == 4:
    # Image 4: Low-key reference (mean=75, std=35) to suppress streetlamp flare
    ref = create_gaussian_reference(
        img.shape, mean=75, std=35, low_val=5, high_val=240
    )
  elif idx == 5:
    # Image 5: Medical soft-tissue target (mean=110, std=35)
    ref = create_gaussian_reference(img.shape, mean=110, std=35)

  hm_img = compute_hm(img, ref)
  results.append((img, he_img, hm_img, idx))

# 4. Visualization: Original vs. HE vs. HM (Images and Histograms)
for orig, he, hm, num in results:
  fig, axes = plt.subplots(2, 3, figsize=(15, 8))

  # Image Plots
  axes[0, 0].imshow(orig, cmap="gray")
  axes[0, 0].set_title(f"Image {num}: Original")
  axes[0, 0].axis("off")

  axes[0, 1].imshow(he, cmap="gray")
  axes[0, 1].set_title(f"Image {num}: Histogram Equalization")
  axes[0, 1].axis("off")

  axes[0, 2].imshow(hm, cmap="gray")
  axes[0, 2].set_title(f"Image {num}: Histogram Matching")
  axes[0, 2].axis("off")

  # Histogram Profiles
  axes[1, 0].hist(orig.ravel(), bins=256, range=[0, 256], color="black")
  axes[1, 0].set_title("Original Histogram")
  axes[1, 0].set_xlim([0, 256])

  axes[1, 1].hist(he.ravel(), bins=256, range=[0, 256], color="blue")
  axes[1, 1].set_title("Equalized Histogram (Flattened CDF)")
  axes[1, 1].set_xlim([0, 256])

  axes[1, 2].hist(hm.ravel(), bins=256, range=[0, 256], color="darkgreen")
  axes[1, 2].set_title("Matched Histogram (Specified Profile)")
  axes[1, 2].set_xlim([0, 256])

  plt.tight_layout()
  plt.show()