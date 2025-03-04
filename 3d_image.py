import numpy as np
import matplotlib.pyplot as plt

# Path to the IMG file
file_path = "source/test/o_01867_optim_no-wind_depth_resamp.img"

# Read raw data (Unsigned Byte)
data = np.fromfile(file_path, dtype=np.uint8)

# Reshape as (Channel, Lines, Samples)
data = data.reshape((2, 4096, 11221))

# Apply scaling and offset from XML
data = (data * 0.4) + 40

# Plot the first channel
plt.figure(figsize=(12, 6))
plt.imshow(data[0], cmap="gray", aspect="auto")
plt.title("MARSIS Radargram (Channel 1)")
plt.colorbar(label="Radar Backscatter Power (dB)")
plt.show()
