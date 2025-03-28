import os
import glob
import dask.dataframe as dd
import pandas as pd
import logging
from datetime import datetime
import argparse
import shutil

# Argument parser
parser = argparse.ArgumentParser(description="Consolidate Mars Express Parquet files")
parser.add_argument("--force", action="store_true", help="Force regeneration of intermediate files")
args = parser.parse_args()

# Logging setup
log_file = f"consolidate_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(filename=log_file, level=logging.INFO, format="%(asctime)s - %(message)s")

def consolidate_intermediate_masters(group_dir):
    """Create intermediate ####_master.parquet files from o_#####_filtered.parquet."""
    subdirs = [d for d in os.listdir(group_dir) if os.path.isdir(d) and d.isdigit()]
    
    for subdir in subdirs:
        full_subdir = os.path.join(group_dir, subdir)
        orbit_dirs = [d for d in os.listdir(full_subdir) if os.path.isdir(os.path.join(full_subdir, d)) and d.startswith("o_")]
        filtered_files = []
        
        for orbit_dir in orbit_dirs:
            pattern = os.path.join(full_subdir, orbit_dir, "o_*_filtered.parquet")
            files = glob.glob(pattern)
            filtered_files.extend(files)
            logging.info(f"Checking {pattern} - Found {len(files)} files: {files}")
        
        if not filtered_files:
            logging.warning(f"No filtered Parquets found in {full_subdir} subdirs")
            continue
        
        # Move output_file into the subdir
        output_file = os.path.join(full_subdir, f"{subdir}_master.parquet")
        if os.path.exists(output_file):
            if args.force:
                if os.path.isdir(output_file):
                    shutil.rmtree(output_file)
                else:
                    os.remove(output_file)
                logging.info(f"Removed existing {output_file} due to --force")
            else:
                logging.info(f"Skipping {output_file} - already exists (use --force to overwrite)")
                continue
        
        logging.info(f"Processing {len(filtered_files)} files for {subdir}")
        df = dd.read_parquet(filtered_files)
        pd_df = df.compute()
        pd_df.to_parquet(output_file, compression="snappy", engine="pyarrow")
        logging.info(f"Created {output_file} with {len(pd_df)} rows")

def consolidate_group_master(group_dir):
    """Combine ####_master.parquet files into ####group_master.parquet."""
    # Update pattern to find masters in subdirs
    pattern = os.path.join(group_dir, "*", "*_master.parquet")
    master_files = glob.glob(pattern)
    logging.info(f"Checking {pattern} - Found {len(master_files)} files: {master_files}")
    
    if not master_files:
        logging.warning("No intermediate masters found")
        return
    
    group_name = os.path.basename(group_dir)
    output_file = os.path.join(group_dir, f"{group_name}_master.parquet")
    if os.path.exists(output_file):
        if args.force:
            if os.path.isdir(output_file):
                shutil.rmtree(output_file)
            else:
                os.remove(output_file)
            logging.info(f"Removed existing {output_file} due to --force")
        else:
            logging.info(f"Skipping {output_file} - already exists (use --force to overwrite)")
            return
    
    logging.info(f"Combining {len(master_files)} intermediate masters")
    df = dd.read_parquet(master_files)
    pd_df = df.compute()
    pd_df.to_parquet(output_file, compression="snappy", engine="pyarrow")
    logging.info(f"Created {output_file} with {len(pd_df)} rows")

if __name__ == "__main__":
    group_dir = os.getcwd()
    logging.info(f"Starting consolidation in {group_dir}")
    
    consolidate_intermediate_masters(group_dir)
    consolidate_group_master(group_dir)
    logging.info("Consolidation complete")

# Notes-to-Self:
# 2025-03-28: Files created but misplaced—all in 1000group/.
# Fixed intermediate output_file to 1000group/1800/1800_master.parquet (was 1000group/1800_master.parquet).
# Updated group master pattern to 1000group/*/*_master.parquet to find intermediates in subdirs.
# Last run worked but wrong locations—now testing correct placement.
# Next: Run with --force, check log for file paths, verify 1800_master.parquet in 1800/, 1000group_master.parquet in 1000group/.
# If memory spikes with compute(), consider Dask alternatives for full 1.5TB later.