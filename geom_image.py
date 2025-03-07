import pandas as pd

# Define file paths
input_file = "source/test/o_01867_geom.tab"  # Path to the .tab file
output_file = "source/test/o_01867_geom.parquet"  # Path to save the Parquet file

# Define column names based on XML metadata
column_names = [
    "Frame", "Ephemeris Time", "Time", "Latitude", "Longitude", "Altitude", 
    "SZA", "Channel 1", "Channel 2", "X", "Y", "Z", "Radial velocity", "Tangential velocity"
]

# Read the .tab file (comma-delimited)
df = pd.read_csv(input_file, delimiter=",", names=column_names, skiprows=0)

# Save as Parquet
df.to_parquet(output_file, engine="pyarrow", compression="snappy", index=False)

print(f"Conversion complete! Parquet file saved as: {output_file}")