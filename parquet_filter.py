import numpy as np
import pandas as pd

# Define file paths
img_file = "source/test/o_01867_optim_no-wind_depth_resamp.img"
parquet_file = "source/test/o_01867_filtered_auto.parquet"

# Define dimensions (from XML)
num_channels = 2
num_lines = 4096  # Total depth layers
num_samples = 11221  # Horizontal distance

# Read raw binary data (Unsigned Byte)
data = np.fromfile(img_file, dtype=np.uint8).reshape((num_channels, num_lines, num_samples))

# Compute dB values (applying scaling and offset from XML)
computed_data = (data * 0.4) + 40  # Converts raw power to dB

# Detect ground level for each sample by finding the first strong radar return
threshold_dB = 60  # Typical surface return threshold (adjustable)
ground_levels = np.argmax(computed_data[0] > threshold_dB, axis=0)  # First strong return per sample

# Determine a global threshold (e.g., 90th percentile of ground detections)
min_subsurface_line = int(np.percentile(ground_levels, 90))  # Conservative depth cutoff

print(f"Automatically detected ground level at line index: {min_subsurface_line}")

# Create DataFrame with filtered data
df = pd.DataFrame({
    "Channel": np.repeat([1, 2], (num_lines - min_subsurface_line) * num_samples),
    "Line": np.tile(np.repeat(np.arange(min_subsurface_line, num_lines), num_samples), num_channels),
    "Sample": np.tile(np.arange(num_samples), (num_lines - min_subsurface_line) * num_channels),
    "Raw Power": data[:, min_subsurface_line:, :].flatten(),
    "Computed Power (dB)": computed_data[:, min_subsurface_line:, :].flatten(),
    "Filtered Depth": (num_lines - np.tile(np.repeat(np.arange(min_subsurface_line, num_lines), num_samples), num_channels))
})

# Save to Parquet with Snappy compression
df.to_parquet(parquet_file, engine="pyarrow", compression="snappy")

print(f"Filtered Parquet file saved: {parquet_file}")
