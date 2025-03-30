import os
import shutil
import argparse
import logging
from datetime import datetime
import numpy as np
import pandas as pd
import cudf  # For single file write
import dask.dataframe as dd
from pathlib import Path
import psutil

# Logging setup
logging.basicConfig(
    filename=f"filter_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def log_memory():
    """Log current memory usage."""
    mem = psutil.virtual_memory()
    print(f"Memory: {mem.used / 1024**3:.2f}/{mem.total / 1024**3:.2f} GB ({mem.percent}%)")
    logging.info(f"Memory: {mem.used / 1024**3:.2f}/{mem.total / 1024**3:.2f} GB ({mem.percent}%)")

def filter_percentile(orbit_dir, orbit, percentile=95, force=False):
    """Filter orbit Parquet to keep rows above the given percentile of Raw_Power."""
    input_path = orbit_dir / f"o_{orbit}_combined.parquet"
    output_path = orbit_dir / f"o_{orbit}_filtered.parquet"
    
    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        logging.warning(f"Input file not found: {input_path}")
        return None
    
    if not force and output_path.exists() and output_path.stat().st_size > 0:
        print(f"Skipping existing Parquet: {output_path}")
        logging.info(f"Skipping existing Parquet: {output_path}")
        return output_path
    if force and output_path.exists():
        if output_path.is_dir():
            shutil.rmtree(output_path)
            print(f"Force deleted directory: {output_path}")
            logging.info(f"Force deleted directory: {output_path}")
        else:
            os.remove(output_path)
            print(f"Force deleted file: {output_path}")
            logging.info(f"Force deleted file: {output_path}")

    try:
        log_memory()
        # Load combined Parquet with Dask
        df = dd.read_parquet(input_path, blocksize="64MB")
        log_memory()

        # Compute 95th percentile of Raw_Power
        percentile_value = df["Raw_Power"].quantile(percentile / 100).compute()
        print(f"95th percentile Raw_Power: {percentile_value}")
        logging.info(f"95th percentile Raw_Power: {percentile_value}")
        log_memory()

        # Filter to keep rows above percentile
        filtered_df = df[df["Raw_Power"] >= percentile_value]
        log_memory()

        # Compute to cuDF and write single file
        filtered_df = filtered_df.compute()  # To Pandas
        filtered_df = cudf.from_pandas(filtered_df)  # To cuDF
        filtered_df.to_parquet(output_path, index=False)  # Single file
        print(f"Filtered orbit {orbit} to {output_path}, shape: {filtered_df.shape}")
        logging.info(f"Filtered orbit {orbit} to {output_path}, shape: {filtered_df.shape}")
        log_memory()
        return output_path
    except Exception as e:
        print(f"Error filtering orbit {orbit}: {e}")
        logging.error(f"Error filtering orbit {orbit}: {e}")
        raise

def main(force=False, percentile=95):
    root = Path.cwd()
    print(f"Scanning current directory and subdirectories: {root}")
    logging.info(f"Scanning current directory and subdirectories: {root}")

    # If --force, delete all existing o_*_filtered.parquet files in subdirectories
    if force:
        filtered_files = list(root.glob("**/o_*_filtered.parquet"))
        for filtered_file in filtered_files:
            if filtered_file.is_file():
                os.remove(filtered_file)
                print(f"Force deleted existing filtered Parquet: {filtered_file}")
                logging.info(f"Force deleted existing filtered Parquet: {filtered_file}")

    # Find all o_*_combined.parquet files recursively
    orbits = set()
    for combined_file in root.glob("**/o_*_combined.parquet"):
        parts = combined_file.name.split('_')
        if len(parts) < 2 or not parts[1].isdigit():
            continue
        orbit = parts[1][:5]
        orbits.add((orbit, combined_file.parent))

    if not orbits:
        print("No combined Parquet files found in current directory or subdirectories.")
        logging.warning("No combined Parquet files found in current directory or subdirectories.")
        return

    for orbit, orbit_dir in sorted(orbits):
        filter_percentile(orbit_dir, orbit, percentile, force)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter orbit Parquet files to 95th percentile of Raw_Power.")
    parser.add_argument("--force", action="store_true", help="Force regeneration")
    parser.add_argument("--percentile", type=float, default=95, help="Percentile to filter on (default: 95)")
    args = parser.parse_args()
    main(args.force, args.percentile)