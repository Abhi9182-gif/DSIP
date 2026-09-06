import cv2
import matplotlib.pyplot as plt
import numpy as np


def contrast_stretch(img):
  r_min, r_max = np.min(img), np.max(img)
  if r_max == r_min:
    return img
  return np.uint8(((img - r_min) / (r_max - r_min)) * 255.0)


def gamma_transform(img, gamma=0.5):
  inv_gamma = gamma
  table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(
      'uint8'
  )
  return cv2.LUT(img, table)


def log_transform(img):
  c = 255.0 / np.log(1.0 + np.max(img))
  log_img = c * np.log(1.0 + img.astype(np.float64))
  return np.uint8(np.clip(log_img, 0, 255))


def apply_clahe(img, clip_limit=2.0, tile_grid_size=(8, 8)):
  clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
  return clahe.apply(img)


# Processing Pipeline
# Note: Load image paths corresponding to the 5 degraded inputs
image_paths = ['img1.png', 'img2.png', 'img3.png', 'img4.png', 'img5.png']

results = []
for i, path in enumerate(image_paths, 1):
  img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
  if img is None:
    continue

  if i == 1:
    # Image 1: Washed out pollen -> Contrast Stretch + Hist Equalization
    stretched = contrast_stretch(img)
    enhanced = cv2.equalizeHist(stretched)
    method = 'Contrast Stretch + Hist Equalization'
  elif i == 2:
    # Image 2: Radial Texture -> CLAHE + Gamma (1.2)
    clahe_out = apply_clahe(img, clip_limit=2.0)
    enhanced = gamma_transform(clahe_out, gamma=1.2)
    method = 'CLAHE + Gamma (1.2)'
  elif i == 3:
    # Image 3: Dark Pollen -> Gamma (0.45) + CLAHE
    gamma_out = gamma_transform(img, gamma=0.45)
    enhanced = apply_clahe(gamma_out, clip_limit=2.5)
    method = 'Gamma (0.45) + CLAHE'
  elif i == 4:
    # Image 4: Dark Car -> Log Transform + CLAHE
    log_out = log_transform(img)
    enhanced = apply_clahe(log_out, clip_limit=3.0)
    method = 'Log Transform + CLAHE'
  elif i == 5:
    # Image 5: Medical CT -> Soft-Tissue Window Slicing + CLAHE
    # Window: Level = 90, Width = 120 -> [30, 150]
    r_min, r_max = 30, 150
    windowed = np.clip((img - r_min) * (255.0 / (r_max - r_min)), 0, 255).astype(
        np.uint8
    )
    enhanced = apply_clahe(windowed, clip_limit=1.8)
    method = 'Dynamic Window Slicing + CLAHE'

  results.append((img, enhanced, method))

# Visualizing Original vs. Enhanced
plt.figure(figsize=(14, 18))
for idx, (orig, enh, title) in enumerate(results):
  plt.subplot(5, 2, 2 * idx + 1)
  plt.imshow(orig, cmap='gray')
  plt.title(f'Original Image {idx+1}')
  plt.axis('off')

  plt.subplot(5, 2, 2 * idx + 2)
  plt.imshow(enh, cmap='gray')
  plt.title(f'Enhanced ({title})')
  plt.axis('off')

plt.tight_layout()
plt.show()