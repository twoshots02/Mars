import pandas as pd

# Load the corrected dataset with the new Depth Below Ground field
parquet_file = "source/test/o_01867_corrected.parquet"
df = pd.read_parquet(parquet_file)

# Compute max depth per sample **ONLY using Depth Below Ground (m)**
max_depth_per_sample = df.groupby("Sample")["Depth Below Ground (m)"].max()

# Check if all max depths are truly consistent across samples
if max_depth_per_sample.nunique() == 1:
    print(f"✅ Max depth is correctly uniform across all samples: {max_depth_per_sample.iloc[0]:.2f} meters")
else:
    print("❌ Max depth still varies across samples!")
    print(max_depth_per_sample.describe())  # Show stats on depth variation
