import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

start_time = time.time()

img_file = "source/test/o_01867_optim_no-wind_depth_resamp.img"
parquet_file = "source/test/o_01867_corrected.parquet"

num_channels = 2
num_lines = 4096
num_samples = 11221
depth_per_pixel = 26.767

data_start = time.time()
data = np.fromfile(img_file, dtype=np.uint8).reshape((num_channels, num_lines, num_samples))
computed_data = (data * 0.4) + 40
data_time = time.time() - data_start

ground_start = time.time()
threshold_dB = 60
ground_levels = np.argmax(computed_data[0] > threshold_dB, axis=0)
ground_levels = np.where(ground_levels == 0, np.argmax(computed_data[0] > 40.4, axis=0), ground_levels)
ground_levels = np.where(ground_levels == 0, 0, ground_levels)
print(f"Detected ground levels range: {ground_levels.min()} to {ground_levels.max()}")
ground_time = time.time() - ground_start

build_start = time.time()
channels_list = []
lines_list = []
samples_list = []
computed_power_list = []
depth_below_ground_list = []

for channel in range(num_channels):
    for sample in range(num_samples):
        if sample % 1000 == 0:
            print(f"Processing Channel {channel + 1}, Sample {sample}")
        ground_line = ground_levels[sample]
        column_data = computed_data[channel, ground_line:, sample]
        if np.any(column_data > 41):
            last_reflection_idx = np.max(np.where(column_data > 41)[0])
            last_line = ground_line + last_reflection_idx
        else:
            continue
        
        subsurface_lines = np.arange(ground_line, last_line + 1)
        num_subsurface_lines = len(subsurface_lines)
        
        channels_list.extend([channel + 1] * num_subsurface_lines)
        samples_list.extend([sample] * num_subsurface_lines)
        lines_list.extend(subsurface_lines)
        
        computed_power_list.extend(computed_data[channel, ground_line:last_line + 1, sample])
        depth_below_ground_list.extend((subsurface_lines - ground_line) * depth_per_pixel)

df = pd.DataFrame({
    "Channel": channels_list,
    "Line": lines_list,
    "Sample": samples_list,
    "Computed Power (dB)": computed_power_list,
    "Depth Below Ground (m)": depth_below_ground_list
})
build_time = time.time() - build_start

save_start = time.time()
df.to_parquet(parquet_file, engine="pyarrow", compression="snappy")
save_time = time.time() - save_start

total_time = time.time() - start_time

print(f"✅ Corrected dataset saved: {parquet_file}")
print(f"DataFrame shape: {df.shape}")
print(df[df['Sample'] == 1].head(10))  # Check a sample with data
print(f"\nTiming Breakdown:")
print(f"  Data Loading and Conversion: {data_time:.2f} seconds")
print(f"  Ground Detection: {ground_time:.2f} seconds")
print(f"  DataFrame Building: {build_time:.2f} seconds")
print(f"  Parquet Save: {save_time:.2f} seconds")
print(f"  Total Time: {total_time:.2f} seconds ({total_time / 60:.2f} minutes)")

# Debug: Plot Sample 0, Channel 0 (non-blocking)
plt.plot(computed_data[0, :, 0], label="Channel 1, Sample 0")
plt.axhline(60, color="r", linestyle="--", label="Threshold 60 dB")
plt.axhline(40.4, color="g", linestyle="--", label="Noise Floor 40.4 dB")
plt.xlabel("Line (Depth)")
plt.ylabel("Computed Power (dB)")
plt.legend()
plt.ion()  # Interactive mode, non-blocking
plt.show()