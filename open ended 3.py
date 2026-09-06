import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import correlate, resample
from scipy.stats import pearsonr
import soundfile as sf

# 1. Folder Setup and Explicit File Selection
folder = r"C:\Users\admin\Documents\DSIP\Lab Experiment"

# Change these 3 file names to match the exact files in your folder:
original_file = "input_song.wav"  # Original song (vocals + music)
karaoke_file = "Gotye (Instrumental).mp3"  # Karaoke/instrumental version
different_file = "different_song.mp3"  # A completely different track

# Verify existence of all 3 files
selected_files = {
    "Original": original_file,
    "Karaoke": karaoke_file,
    "Different": different_file,
}

for label, filename in selected_files.items():
  full_path = os.path.join(folder, filename)
  if not os.path.isfile(full_path):
    raise FileNotFoundError(
        f"Could not find {label} track: '{filename}'\n"
        f"Make sure '{filename}' exists in:\n{folder}"
    )

print("All 3 required audio files successfully located:")
for label, filename in selected_files.items():
  print(f" - {label:9s}: {filename}")


# 2. Audio Loading & Preprocessing Function (Mono + Resample to 8000 Hz)
def load_and_preprocess(filepath, target_sr=8000):
  data, sr = sf.read(filepath)

  # Convert stereo (or multi-channel) to mono by averaging channels
  if data.ndim > 1:
    data = np.mean(data, axis=1)

  # Resample to common target sample rate (8000 Hz)
  if sr != target_sr:
    num_samples = int(len(data) * target_sr / sr)
    data = resample(data, num_samples)

  # Z-score standardization: zero mean, unit variance
  data = (data - np.mean(data)) / np.std(data)
  return data


print("\nLoading and preprocessing audio signals...")
s_orig = load_and_preprocess(os.path.join(folder, original_file))
s_kara = load_and_preprocess(os.path.join(folder, karaoke_file))
s_diff = load_and_preprocess(os.path.join(folder, different_file))

# Truncate all audio arrays to the shortest length
min_len = min(len(s_orig), len(s_kara), len(s_diff))
s_orig = s_orig[:min_len]
s_kara = s_kara[:min_len]
s_diff = s_diff[:min_len]

print(
    f"Audio arrays synchronized to common length: {min_len} samples"
    f" ({min_len / 8000:.2f} seconds)"
)

# 3. Pearson Correlation (Zero-lag instantaneous comparison)
r_ok, _ = pearsonr(s_orig, s_kara)
r_od, _ = pearsonr(s_orig, s_diff)
r_kd, _ = pearsonr(s_kara, s_diff)


# 4. Normalized Cross-Correlation (Lag-dependent comparison)
def normalized_cross_correlation(a, b):
  result = correlate(a, b, mode="full")
  denominator = np.linalg.norm(a) * np.linalg.norm(b)
  return result / denominator


print("Computing cross-correlations across all lags...")
cross_ok = normalized_cross_correlation(s_orig, s_kara)
cross_od = normalized_cross_correlation(s_orig, s_diff)
cross_kd = normalized_cross_correlation(s_kara, s_diff)

max_ok = np.max(np.abs(cross_ok))
max_od = np.max(np.abs(cross_od))
max_kd = np.max(np.abs(cross_kd))

# 5. Display Statistical Metrics
print("\n========================================")
print("          PEARSON CORRELATION")
print("========================================")
print(f"Original vs Karaoke   : {r_ok:.4f}")
print(f"Original vs Different : {r_od:.4f}")
print(f"Karaoke vs Different  : {r_kd:.4f}")

print("\n========================================")
print("     MAX NORMALIZED CROSS-CORRELATION")
print("========================================")
print(f"Original vs Karaoke   : {max_ok:.4f}")
print(f"Original vs Different : {max_od:.4f}")
print(f"Karaoke vs Different  : {max_kd:.4f}")

# 6. Plotting Cross-Correlation Curves
lags = np.arange(-min_len + 1, min_len)

plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(lags, cross_ok, color="blue", linewidth=0.8)
plt.title(f"Original vs Karaoke (Peak: {max_ok:.4f})")
plt.xlabel("Sample Lag")
plt.ylabel("Correlation")
plt.grid(True, alpha=0.3)

plt.subplot(3, 1, 2)
plt.plot(lags, cross_od, color="red", linewidth=0.8)
plt.title(f"Original vs Different Song (Peak: {max_od:.4f})")
plt.xlabel("Sample Lag")
plt.ylabel("Correlation")
plt.grid(True, alpha=0.3)

plt.subplot(3, 1, 3)
plt.plot(lags, cross_kd, color="green", linewidth=0.8)
plt.title(f"Karaoke vs Different Song (Peak: {max_kd:.4f})")
plt.xlabel("Sample Lag")
plt.ylabel("Correlation")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 7. Plotting Pearson Bar Chart
labels = ["Original-Karaoke", "Original-Different", "Karaoke-Different"]
values = [r_ok, r_od, r_kd]

plt.figure(figsize=(8, 5))
bars = plt.bar(
    labels,
    values,
    color=["royalblue", "salmon", "mediumseagreen"],
    edgecolor="black",
)
plt.axhline(0, color="black", linestyle="--", linewidth=0.8)
plt.title("Pearson Correlation Between Audio Tracks")
plt.ylabel("Correlation Coefficient (r)")
plt.grid(axis="y", alpha=0.3)

for bar in bars:
  height = bar.get_height()
  va = "bottom" if height >= 0 else "top"
  plt.annotate(
      f"{height:.4f}",
      xy=(bar.get_x() + bar.get_width() / 2, height),
      xytext=(0, 3 if height >= 0 else -3),
      textcoords="offset points",
      ha="center",
      va=va,
      fontsize=10,
  )

plt.tight_layout()
plt.show()