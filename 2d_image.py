import numpy as np
import matplotlib.pyplot as plt

# Path to the IMG file in the "source/test" subdirectory
file_path = "source/test/o_01867_MOLA_shaded_resamp.img"

# Read the raw data into a NumPy array (UInt16, little-endian)
data = np.fromfile(file_path, dtype=np.uint16)
data = data.reshape((800, 11220))

# Display the image using matplotlib
plt.figure(figsize=(12, 4))
plt.imshow(data, cmap='gray')
plt.title("MOLA Shaded Relief Image")
plt.colorbar(label="Pixel Value")
plt.show()
