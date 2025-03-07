import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import psutil

# Function to print memory usage
def print_memory_usage():
    process = psutil.Process()
    print(f"Memory Usage: {process.memory_info().rss / 1024 ** 2:.2f} MB")

# Load Parquet files
radar_df = pd.read_parquet("source/test/o_01867_parallel.parquet")
mola_df = pd.read_parquet("source/test/o_01867_mola_resamp.parquet")
geom_df = pd.read_parquet("source/test/o_01867_geom.parquet")

# Debug: Inspect DataFrames
print("Radar DataFrame columns:", radar_df.columns)
print("Radar DataFrame shape:", radar_df.shape)
print_memory_usage()
print("MOLA DataFrame columns:", mola_df.columns)
print("MOLA DataFrame shape:", mola_df.shape)
print_memory_usage()
print("Geometry DataFrame columns:", geom_df.columns)
print("Geometry DataFrame shape:", geom_df.shape)
print_memory_usage()

# Step 1: Prepare Geometry DataFrame for interpolation
geom_df["Time"] = pd.to_datetime(geom_df["Time"])
geom_df = geom_df.set_index("Time")

for col in ["Latitude", "Longitude", "Altitude", "SZA", "X", "Y", "Z", "Radial velocity", "Tangential velocity", "Ephemeris Time", "Frame"]:
    geom_df[col] = pd.to_numeric(geom_df[col], errors="coerce")

time_range = pd.date_range(start="2005-06-29T12:28:20.947Z", end="2005-06-29T12:54:32.117Z", periods=11221)
geom_interp = geom_df.reindex(time_range).interpolate(method="linear")
geom_interp["Sample"] = np.arange(11221)
print("Geometry Interpolation completed. Shape:", geom_interp.shape)
print_memory_usage()

# Step 2: Prepare MOLA DataFrame
mola_pivot = mola_df.pivot(index="Sample", columns="Line", values="Elevation")
mola_pivot.columns = [f"MOLA_Elevation_{int(col)}" for col in mola_pivot.columns]
print("MOLA Pivot created. Shape:", mola_pivot.shape)
print_memory_usage()

if mola_pivot.shape[0] == 11220:
    mola_pivot = mola_pivot.reindex(np.arange(11221)).ffill()
    print("MOLA samples aligned to 11,221.")
print("MOLA Pivot DataFrame columns:", mola_pivot.columns)
print("MOLA Elevation Range (m):", mola_pivot["MOLA_Elevation_0"].min(), "to", mola_pivot["MOLA_Elevation_0"].max())
print_memory_usage()

# Step 3: Aggregate radar_df per sample, picking depth at max power
# Filter radar_df to only include meaningful reflections (> 40 dB)
radar_filtered = radar_df[radar_df["Computed Power (dB)"] > 40]
print("Radar filtered (power > 40 dB). Shape:", radar_filtered.shape)
print("Radar filtered power range (dB):", radar_filtered["Computed Power (dB)"].min(), "to", radar_filtered["Computed Power (dB)"].max())

# Find the depth at the maximum power for each sample
radar_agg = radar_filtered.groupby("Sample").apply(
    lambda x: x.loc[x["Computed Power (dB)"].idxmax(), ["Computed Power (dB)", "Depth Below Ground (m)"]]
    if not x.empty else pd.Series({"Computed Power (dB)": 40.0, "Depth Below Ground (m)": 0.0})
).reset_index()
print("Radar aggregated (depth at max power). Shape:", radar_agg.shape)
print("Radar aggregated power range (dB):", radar_agg["Computed Power (dB)"].min(), "to", radar_agg["Computed Power (dB)"].max())
print("Radar aggregated depth range (m):", radar_agg["Depth Below Ground (m)"].min(), "to", radar_agg["Depth Below Ground (m)"].max())
print_memory_usage()

# Step 4: Merge DataFrames
combined_df = pd.merge(radar_agg, geom_interp, on="Sample", how="left")
print("Merged radar with geometry. Shape:", combined_df.shape)
print_memory_usage()

combined_df = pd.merge(combined_df, mola_pivot, left_on="Sample", right_index=True, how="left")
print("Merged with MOLA. Shape:", combined_df.shape)
print_memory_usage()

# Step 5: Calculate Depth Above Ellipsoid
print("Checking for NaN in MOLA_Elevation_0:", combined_df["MOLA_Elevation_0"].isna().sum())
print("Checking for NaN in Depth Below Ground (m):", combined_df["Depth Below Ground (m)"].isna().sum())
combined_df["Depth Above Ellipsoid (m)"] = combined_df["MOLA_Elevation_0"] - combined_df["Depth Below Ground (m)"]
print("Depth Above Ellipsoid calculated.")
print("Depth Above Ellipsoid Range (m):", combined_df["Depth Above Ellipsoid (m)"].min(), "to", combined_df["Depth Above Ellipsoid (m)"].max())
print_memory_usage()

# Debug: Print combined DataFrame
print("Combined DataFrame columns:", combined_df.columns)
print(combined_df[["Sample", "Computed Power (dB)", "Depth Below Ground (m)", "MOLA_Elevation_0", "Depth Above Ellipsoid (m)"]].head())

# Step 6: Visualization
# Plot depth below ground per sample
plt.figure(figsize=(12, 5))
plt.plot(combined_df["Sample"], combined_df["Depth Below Ground (m)"], label="Depth at Max Power per Sample", color="b")
plt.axhline(y=combined_df["Depth Below Ground (m)"].mean(), color="r", linestyle="--", label=f"Mean Depth ({combined_df['Depth Below Ground (m)'].mean():.2f} m)")
plt.axhline(y=5000, color="g", linestyle="--", label="MARSIS Limit (5 km)")
plt.xlabel("Sample Index")
plt.ylabel("Depth Below Ground (m)")
plt.title("Depth Below Ground Per Sample (at Max Power)")
plt.legend()
plt.show()

# Depth histogram
plt.figure(figsize=(8, 5))
plt.hist(combined_df["Depth Below Ground (m)"].dropna(), bins=50, color="b", alpha=0.7)
plt.axvline(x=5000, color="g", linestyle="--", label="MARSIS Limit (5 km)")
plt.xlabel("Depth Below Ground (m)")
plt.ylabel("Frequency")
plt.title("Depth Distribution")
plt.legend()
plt.show()