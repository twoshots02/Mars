import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing as mp
import time

start_time = time.time()

img_file = "source/test/o_01867_optim_no-wind_depth_resamp.img"
parquet_file = "source/test/o_01867_parallel.parquet"

num_channels = 2
num_lines = 4096
num_samples = 11221
depth_per_pixel = 26.767

data_start = time.time()
data = np.fromfile(img_file, dtype=np.uint8).reshape((num_channels, num_lines, num_samples))
computed_data = (data * 0.4) + 40
data_time = time.time() - data_start

ground_start = time.time()
threshold_dB = 50
ground_levels = np.argmax(computed_data[0] > threshold_dB, axis=0)
ground_levels = np.where(ground_levels == 0, np.argmax(computed_data[0] > 40.4, axis=0), ground_levels)
ground_levels = np.where(ground_levels == 0, 0, ground_levels)
print(f"Detected ground levels range: {ground_levels.min()} to {ground_levels.max()}")
ground_time = time.time() - ground_start

def process_sample(sample):
    sample_data = []
    for channel in range(num_channels):
        ground_line = ground_levels[sample]
        if ground_line >= num_lines:
            continue
        column_data = computed_data[channel, ground_line:, sample]
        if not np.any(column_data > 45):  # Increased to 45 dB
            continue
        print(f"Processing Sample {sample}, Channel {channel + 1}, Ground Line {ground_line}")  # Debug
        last_reflection_idx = np.max(np.where(column_data > 45)[0])
        last_line = ground_line + last_reflection_idx
        
        subsurface_lines = np.arange(ground_line, last_line + 1)
        for line in subsurface_lines:
            depth_below_ground = (line - ground_line) * depth_per_pixel
            if depth_below_ground > 5000:
                continue
            sample_data.append((
                channel + 1,
                line,
                sample,
                computed_data[channel, line, sample],
                depth_below_ground
            ))
    return sample_data

build_start = time.time()
with mp.Pool(processes=min(mp.cpu_count(), 8)) as pool:
    results = pool.map(process_sample, range(num_samples))

flattened_results = [row for sample in results for row in sample]
df = pd.DataFrame(flattened_results, columns=["Channel", "Line", "Sample", "Computed Power (dB)", "Depth Below Ground (m)"])
build_time = time.time() - build_start

save_start = time.time()
df.to_parquet(parquet_file, engine="pyarrow", compression="snappy")
save_time = time.time() - save_start

total_time = time.time() - start_time

print(f"✅ Parallelized dataset saved: {parquet_file}")
print(f"DataFrame shape: {df.shape}")
print(df[df['Sample'] == 1].head(10))  # Check Sample 1
print(f"\nTiming Breakdown:")
print(f"  Data Loading and Conversion: {data_time:.2f} seconds")
print(f"  Ground Detection: {ground_time:.2f} seconds")
print(f"  DataFrame Building (Parallelized): {build_time:.2f} seconds")
print(f"  Parquet Save: {save_time:.2f} seconds")
print(f"  Total Time: {total_time:.2f} seconds ({total_time / 60:.2f} minutes)")

# Visualization
plt.figure(figsize=(12, 5))
max_depths = df.groupby("Sample")["Depth Below Ground (m)"].max()
plt.plot(max_depths, label="Max Depth per Sample", color="b")
plt.axhline(y=max_depths.mean(), color="r", linestyle="--", label=f"Mean Depth ({max_depths.mean():.2f} m)")
plt.axhline(y=5000, color="g", linestyle="--", label="MARSIS Limit (5 km)")
plt.xlabel("Sample Index")
plt.ylabel("Max Depth Below Ground (m)")
plt.title("Max Depth Below Ground Per Sample")
plt.legend()

# Depth histogram
plt.figure(figsize=(8, 5))
plt.hist(df["Depth Below Ground (m)"], bins=50, color="b", alpha=0.7)
plt.axvline(x=5000, color="g", linestyle="--", label="MARSIS Limit (5 km)")
plt.xlabel("Depth Below Ground (m)")
plt.ylabel("Frequency")
plt.title("Depth Distribution")
plt.legend()
plt.show()