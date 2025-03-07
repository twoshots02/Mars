import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Path to the IMG file in the "source/test" subdirectory
file_path = "source/test/o_01867_MOLA_shaded_resamp.img"
output_parquet = "source/test/o_01867_mola_resamp.parquet"

# Read the raw data into a NumPy array (UInt16, little-endian)
data = np.fromfile(file_path, dtype=np.uint16)
data = data.reshape((800, 11220))

# Convert to long-format DataFrame
num_lines, num_samples = data.shape
lines, samples = np.meshgrid(np.arange(num_lines), np.arange(num_samples), indexing='ij')
mola_df = pd.DataFrame({
    "Line": lines.ravel(),
    "Sample": samples.ravel(),
    "Elevation": data.ravel()
})

# Save to Parquet
mola_df.to_parquet(output_parquet, engine="pyarrow", compression="snappy", index=False)
print(f"Parquet file saved as: {output_parquet}")

# Display the image using matplotlib for validation
plt.figure(figsize=(12, 4))
plt.imshow(data, cmap='gray')
plt.title("MOLA Shaded Relief Image")
plt.colorbar(label="Elevation (Pixel Value)")
plt.show()

# Print summary statistics
print(f"Min Elevation: {mola_df['Elevation'].min()}")
print(f"Max Elevation: {mola_df['Elevation'].max()}")
print(f"Shape of DataFrame: {mola_df.shape}")