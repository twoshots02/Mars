import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing as mp
import time

start_time = time.time()

img_file = "source/test/o_01867_optim_no-wind_depth_resamp.img"
parquet_file = "source/test/o_01867_parallel_with_raw.parquet"

num_channels = 2
num_lines = 4096
num_samples = 11221
depth_per_pixel = 26.767

data_start = time.time()
data = np.fromfile(img_file, dtype=np.uint8).reshape((num_channels, num_lines, num_samples))
computed_data = (data * 0.4) + 40  # Convert raw values to dB
data_time = time.time() - data_start

ground_start = time.time()
threshold_dB = 45  # Adjusted threshold for surface detection
ground_levels = np.argmax(computed_data[0] > threshold_dB, axis=0)
ground_levels = np.where(ground_levels == 0, np.argmax(computed_data[0] > 40, axis=0), ground_levels)  
ground_levels = np.where(ground_levels == 0, 0, ground_levels)

# Validation
max_allowed_line = num_lines - 1  # 4095
invalid_ground_indices = np.where(ground_levels > max_allowed_line)[0]
if len(invalid_ground_indices) > 0:
    print(f"Warning: {len(invalid_ground_indices)} ground levels exceed {max_allowed_line}")
    ground_levels[ground_levels > max_allowed_line] = max_allowed_line

print(f"Ground levels range: {ground_levels.min()} to {ground_levels.max()}")
print(f"First 10 ground levels: {ground_levels[:10]}")

plt.plot(ground_levels)
plt.xlabel('Sample Number')
plt.ylabel('Ground Level (Line)')
plt.title('Detected Ground Levels Across Samples')
plt.show()

def process_sample(sample):
    sample_data = []
    for channel in range(num_channels):
        ground_line = ground_levels[sample]
        if ground_line >= num_lines:
            continue
        column_data = computed_data[channel, ground_line:, sample]
        raw_column_data = data[channel, ground_line:, sample]  # Raw power values
        
        max_depth_idx = int(5000 / depth_per_pixel)  
        reflection_indices = np.where(column_data > 40)[0]
        if len(reflection_indices) > 0:
            reflection_indices = reflection_indices[reflection_indices <= max_depth_idx]
        else:
            reflection_indices = [0]  

        subsurface_lines = ground_line + reflection_indices
        
        for i, line in enumerate(subsurface_lines):
            depth_below_ground = (line - ground_line) * depth_per_pixel
            sample_data.append((
                channel + 1,
                line,
                sample,
                raw_column_data[i],  # Raw Power
                computed_data[channel, line, sample],  # Computed dB
                depth_below_ground
            ))

        for line in range(ground_line + 1, min(ground_line + max_depth_idx + 1, num_lines)):
            if line not in subsurface_lines:
                depth_below_ground = (line - ground_line) * depth_per_pixel
                sample_data.append((
                    channel + 1,
                    line,
                    sample,
                    data[channel, line, sample],  # Raw Power
                    computed_data[channel, line, sample],  # Computed dB
                    depth_below_ground
                ))
    return sample_data

build_start = time.time()
with mp.Pool(processes=min(mp.cpu_count(), 8)) as pool:
    results = pool.map(process_sample, range(num_samples))

flattened_results = [row for sample in results for row in sample]
df = pd.DataFrame(flattened_results, columns=["Channel", "Line", "Sample", "Raw Power", "Computed Power (dB)", "Depth Below Ground (m)"])
build_time = time.time() - build_start

save_start = time.time()
df.to_parquet(parquet_file, engine="pyarrow", compression="snappy")
save_time = time.time() - save_start

total_time = time.time() - start_time

print(f"Data processing complete. Parquet file saved: {parquet_file}")
print(f"Total execution time: {total_time:.2f} seconds")
