import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import psutil

# Function to print memory usage
def print_memory_usage():
    process = psutil.Process()
    print(f"Memory Usage: {process.memory_info().rss / 1024 ** 2:.2f} MB")

# Load Parquet files
radar_df = pd.read_parquet("source/test/o_01867_parallel_with_raw.parquet")
mola_df = pd.read_parquet("source/test/o_01867_mola_resamp.parquet")
geom_df = pd.read_parquet("source/test/o_01867_geom.parquet")

# Clean radar_df: Remove Computed Power (dB) if present
if "Computed Power (dB)" in radar_df.columns:
    radar_df = radar_df.drop(columns=["Computed Power (dB)"])
    print("Removed Computed Power (dB) from radar_df.")

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
# Debug: Verify the Time column exists and its values
print("Geometry DataFrame Time column before conversion:", geom_df["Time"].head())
# Convert Time column to datetime with UTC timezone
geom_df["Time"] = pd.to_datetime(geom_df["Time"], utc=True)
print("Geometry DataFrame Time column after conversion:", geom_df["Time"].head())
print("Geometry Time Range before setting index:", geom_df["Time"].min(), "to", geom_df["Time"].max())

# Drop unnecessary geometry columns
geom_df = geom_df.drop(columns=["SZA", "X", "Y", "Z", "Radial velocity", "Tangential velocity", "Ephemeris Time", "Channel 1", "Channel 2"])
print("Dropped unnecessary geometry columns. Remaining columns:", geom_df.columns)

# Set the Time column as the index
geom_df = geom_df.set_index("Time")
# Debug: Verify the index
print("Geometry DataFrame index after set_index:", geom_df.index)
print("Geometry Time Range after setting index:", geom_df.index.min(), "to", geom_df.index.max())

# Clean Latitude, Longitude, Altitude, and Frame
geom_df["Latitude"] = geom_df["Latitude"].astype(str).str.replace(r'[^\d.-]', '', regex=True)
geom_df["Longitude"] = geom_df["Longitude"].astype(str).str.replace(r'[^\d.-]', '', regex=True)
for col in ["Latitude", "Longitude", "Altitude", "Frame"]:
    geom_df[col] = pd.to_numeric(geom_df[col], errors="coerce")
    print(f"{col} NaN count after coercion:", geom_df[col].isna().sum())
    print(f"{col} range: {geom_df[col].min()} to {geom_df[col].max()}")

# Create the time range for interpolation (UTC)
time_range = pd.date_range(start="2005-06-29T12:28:20.947Z", end="2005-06-29T12:54:32.117Z", periods=11221)
print("Constructed Time Range:", time_range[0], "to", time_range[-1])

# Reindex and fill initial NaN values with nearest, then interpolate
geom_interp = geom_df.reindex(time_range, method="nearest").interpolate(method="linear").reset_index(drop=True)
geom_interp["Sample"] = np.arange(11221)
# Fill any remaining NaN at the edges
geom_interp["Latitude"] = geom_interp["Latitude"].ffill().bfill()
geom_interp["Longitude"] = geom_interp["Longitude"].ffill().bfill()
print("Geometry Interpolation completed. Shape:", geom_interp.shape)
print("Geometry Latitude Range:", geom_interp["Latitude"].min(), "to", geom_interp["Latitude"].max())
print("Geometry Longitude Range:", geom_interp["Longitude"].min(), "to", geom_interp["Longitude"].max())
# Debug: Check ranges of remaining geometry columns after interpolation
for col in ["Altitude", "Frame"]:
    print(f"{col} range after interpolation: {geom_interp[col].min()} to {geom_interp[col].max()}")
print_memory_usage()

# Step 2: Prepare MOLA DataFrame (keep only nadir elevation, assuming Line 400)
mola_nadir = mola_df[mola_df["Line"] == 400]  # Select nadir line
mola_pivot = mola_nadir.pivot(index="Sample", columns="Line", values="Elevation")
mola_pivot.columns = [f"MOLA_Elevation_{int(col)}" for col in mola_pivot.columns]
# Scale MOLA elevation from uint16 to meters (-8,000 to +21,000 m)
mola_pivot = (mola_pivot / 65535) * (21000 - (-8000)) + (-8000)
print("MOLA Pivot created (nadir only). Shape:", mola_pivot.shape)
print_memory_usage()

if mola_pivot.shape[0] == 11220:
    mola_pivot = mola_pivot.reindex(np.arange(11221)).ffill()
    print("MOLA samples aligned to 11,221.")
print("MOLA Pivot DataFrame columns:", mola_pivot.columns)
print("MOLA Elevation Range (m):", mola_pivot["MOLA_Elevation_400"].min(), "to", mola_pivot["MOLA_Elevation_400"].max())
print_memory_usage()

# Step 3: Aggregate radar_df per sample, picking depth at max raw power
radar_filtered = radar_df[radar_df["Raw Power"] > 10]
print("Radar filtered (raw power > 10). Shape:", radar_filtered.shape)
print("Radar filtered raw power range:", radar_filtered["Raw Power"].min(), "to", radar_filtered["Raw Power"].max())

radar_agg = radar_filtered.groupby("Sample").apply(
    lambda x: x.loc[x["Raw Power"].idxmax(), ["Raw Power", "Depth Below Ground (m)"]]
    if not x.empty else pd.Series({"Raw Power": 0, "Depth Below Ground (m)": 0.0})
).reset_index()
print("Radar aggregated (depth at max raw power). Shape:", radar_agg.shape)
print("Radar aggregated raw power range:", radar_agg["Raw Power"].min(), "to", radar_agg["Raw Power"].max())
print("Radar aggregated depth range (m):", radar_agg["Depth Below Ground (m)"].min(), "to", radar_agg["Depth Below Ground (m)"].max())
print_memory_usage()

# Step 4: Merge DataFrames
combined_df = pd.merge(radar_agg, geom_interp, on="Sample", how="left")
print("Merged radar with geometry. Shape:", combined_df.shape)
print_memory_usage()

combined_df = pd.merge(combined_df, mola_pivot, left_on="Sample", right_index=True, how="left")
print("Merged with MOLA (nadir only). Shape:", combined_df.shape)
print_memory_usage()

# Step 5: Calculate Depth Above Ellipsoid and Associate Locations
print("Checking for NaN in MOLA_Elevation_400:", combined_df["MOLA_Elevation_400"].isna().sum())
print("Checking for NaN in Depth Below Ground (m):", combined_df["Depth Below Ground (m)"].isna().sum())
combined_df["Depth Above Ellipsoid (m)"] = combined_df["MOLA_Elevation_400"] - combined_df["Depth Below Ground (m)"]
print("Depth Above Ellipsoid calculated.")
print("Depth Above Ellipsoid Range (m):", combined_df["Depth Above Ellipsoid (m)"].min(), "to", combined_df["Depth Above Ellipsoid (m)"].max())
print_memory_usage()

# Associate locations with samples
combined_df["Latitude"] = combined_df["Latitude"].ffill().bfill()
combined_df["Longitude"] = combined_df["Longitude"].ffill().bfill()
print("Location association completed. Sample 0 location:", combined_df.loc[0, ["Latitude", "Longitude"]])

# Debug: Check ranges of all float columns in combined_df
for col in ["Raw Power", "Depth Below Ground (m)", "Frame", "Altitude", "MOLA_Elevation_400", "Depth Above Ellipsoid (m)", "Latitude", "Longitude"]:
    print(f"{col} range in combined_df: {combined_df[col].min()} to {combined_df[col].max()}")

# Save combined DataFrame to Parquet for debugging
output_file = "combined_marsis_orbit_01867_nadir_reduced.parquet"
combined_df.to_parquet(output_file, index=False)
print(f"Combined DataFrame saved to {output_file} for debugging.")

# Debug: Print combined DataFrame
print("Combined DataFrame columns:", combined_df.columns)
print(combined_df[["Sample", "Raw Power", "Depth Below Ground (m)", "MOLA_Elevation_400", "Depth Above Ellipsoid (m)", "Latitude", "Longitude"]].head())

# Step 6: Visualization
# Plot depth below ground per sample
plt.figure(figsize=(12, 5))
plt.plot(combined_df["Sample"], combined_df["Depth Below Ground (m)"], label="Depth at Max Raw Power per Sample", color="b")
plt.axhline(y=combined_df["Depth Below Ground (m)"].mean(), color="r", linestyle="--", label=f"Mean Depth ({combined_df['Depth Below Ground (m)'].mean():.2f} m)")
plt.axhline(y=5000, color="g", linestyle="--", label="MARSIS Limit (5 km)")
plt.xlabel("Sample Index")
plt.ylabel("Depth Below Ground (m)")
plt.title("Depth Below Ground Per Sample (at Max Raw Power)")
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