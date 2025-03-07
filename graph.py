import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the Parquet file
df = pd.read_parquet('source/test/o_01867_parallel.parquet')

# Define constants
num_samples = 11221
num_lines = 4096
depth_per_pixel = 26.767
max_depth = num_lines * depth_per_pixel  # ~109,558 m

# Filter for reflections above threshold
#df = df[df["Computed Power (dB)"] > 40]  # Apply threshold to plotting

# Find global min and max for computed power
computed_power_min = min(40, df["Computed Power (dB)"].min())
computed_power_max = max(120, df["Computed Power (dB)"].max())

def create_plot(df_channel, channel, computed_power_min, computed_power_max, num_lines, depth_per_pixel):
    array = np.full((num_lines, num_samples), np.NaN)
    for _, row in df_channel.iterrows():
        sample = int(row["Sample"])
        line = int(row["Line"])
        computed_power = float(row["Computed Power (dB)"])
        array[line, sample] = computed_power
    
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(array, aspect='auto', interpolation='none', cmap='gray',
                   vmin=computed_power_min, vmax=computed_power_max)
    ax.set_title(f"Channel {channel}")
    ax.set_xlabel('Sample Number')
    ax.set_ylabel('Depth (m)', rotation=90)
    depth_ticks = np.linspace(0, num_lines - 1, 5)
    depth_labels = [f"{int(line * depth_per_pixel)} m" for line in depth_ticks]
    ax.set_yticks(depth_ticks)
    ax.set_yticklabels(depth_labels)
    ax.set_xticks(np.array([i * 1000 for i in range(12)]))
    ax.set_xticklabels(np.array([i * 1000 for i in range(12)]))
    plt.colorbar(im, ax=ax, label='Computed Power (dB)', orientation='horizontal')
    plt.show()

for channel in [1, 2]:
    df_channel = df[df["Channel"] == channel]
    if not df_channel.empty:
        create_plot(df_channel, channel, computed_power_min, computed_power_max, num_lines, depth_per_pixel)
    else:
        print(f"No data available for Channel {channel}")