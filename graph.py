import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the Parquet file
df = pd.read_parquet('source/test/o_01867_parallel.parquet')

# Define constants
num_samples = 11221
depth_per_pixel = 26.767
max_depth = 5000
max_depth_index = int(max_depth / depth_per_pixel)  # 186

# Find global min and max for computed power
computed_power_min = df["Computed Power (dB)"].min()
computed_power_max = df["Computed Power (dB)"].max()

# Function to create the plot for a channel
def create_plot(df_channel, channel, computed_power_min, computed_power_max, max_depth_index, max_depth):
    # Create the 2D array
    array = np.full((max_depth_index + 1, num_samples), np.NaN)
    
    # Fill the array with bounds checking
    for _, row in df_channel.iterrows():
        sample = int(row["Sample"])  # Ensure integer
        depth = float(row["Depth Below Ground (m)"])  # Ensure float for division
        computed_power = float(row["Computed Power (dB)"])  # Ensure float
        
        # Calculate depth index with bounds checking
        depth_index = int(round(depth / depth_per_pixel))
        if 0 <= depth_index <= max_depth_index and 0 <= sample < num_samples:
            array[depth_index, sample] = computed_power
        else:
            print(f"Warning: Out of bounds - Sample: {sample}, Depth Index: {depth_index}")
    
    # Plot the array
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(array, aspect='auto', interpolation='none', cmap='viridis',
                   vmin=computed_power_min, vmax=computed_power_max)
    ax.set_title(f"Channel {channel}")
    ax.set_xlabel('Sample Number')
    ax.set_ylabel('Depth (m)', rotation=90)
    ax.set_yticks(np.linspace(0, max_depth_index, 5))
    ax.set_yticklabels(np.linspace(0, max_depth, 5).astype(int))
    ax.set_xticks(np.array([i * 1000 for i in range(12)]))
    ax.set_xticklabels(np.array([i * 1000 for i in range(12)]))
    plt.colorbar(im, ax=ax, label='Computed Power (dB)')
    plt.show()

# Separate data by channel and create plots
for channel in [1, 2]:
    df_channel = df[df["Channel"] == channel]
    if not df_channel.empty:  # Check if channel has data
        create_plot(df_channel, channel, computed_power_min, computed_power_max, max_depth_index, max_depth)
    else:
        print(f"No data available for Channel {channel}")