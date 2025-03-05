import pandas as pd

# Define file paths
input_file = "source/test/o_01867_geom.tab"  # Path to the .tab file
output_file = "source/test/o_01867_geom.csv"  # Path to save the CSV file

# Define column names based on XML metadata
column_names = [
    "Frame", "Ephemeris Time", "Time", "Latitude", "Longitude", "Altitude", 
    "SZA", "Channel 1", "Channel 2", "X", "Y", "Z", "Radial velocity", "Tangential velocity"
]

# Read the .tab file (comma-delimited)
df = pd.read_csv(input_file, delimiter=",", names=column_names, skiprows=0)

# Save as CSV (without an index column)
df.to_csv(output_file, index=False)

print(f"Conversion complete! CSV saved as: {output_file}")
