import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
"""
This script processes a MOLA shaded relief image, converts it to a long-format DataFrame, 
saves it as a Parquet file, and displays the image using matplotlib.

Steps:
1. Reads the raw data from an IMG file into a NumPy array.
2. Reshapes the data to the specified dimensions.
3. Converts the reshaped data into a long-format pandas DataFrame.
4. Saves the DataFrame to a Parquet file with Snappy compression.
5. Displays the image using matplotlib for validation.
6. Prints summary statistics of the elevation data.

File Paths:
- Input IMG file: "source/test/o_01867_MOLA_shaded_resamp.img"
- Output Parquet file: "source/test/o_01867_mola_resamp.parquet"

Data:
- The raw data is read as UInt16, little-endian.
- The data is reshaped to dimensions (800, 11220).

DataFrame Columns:
- "Line": The line index of the image.
- "Sample": The sample index of the image.
- "Elevation": The elevation value (pixel value) from the image.

Summary Statistics:
- Minimum elevation value.
- Maximum elevation value.
- Shape of the DataFrame.

Visualization:
- Displays the image using a grayscale colormap.
- Adds a colorbar to indicate elevation (pixel value).
"""


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