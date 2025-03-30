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
from xml.etree import ElementTree as ET

# Logging setup
logging.basicConfig(
    filename=f"merge_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def log_memory():
    """Log current memory usage."""
    mem = psutil.virtual_memory()
    print(f"Memory: {mem.used / 1024**3:.2f}/{mem.total / 1024**3:.2f} GB ({mem.percent}%)")
    logging.info(f"Memory: {mem.used / 1024**3:.2f}/{mem.total / 1024**3:.2f} GB ({mem.percent}%)")

def get_img_dims(xml_path, array_type):
    """Extract array dimensions from XML for IMG files."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        ns = {"pds": "http://pds.nasa.gov/pds4/pds/v1"}
        array = root.find(f".//pds:{array_type}", ns)
        if array is None:
            raise ValueError(f"No {array_type} found in {xml_path}")
        axes = [(a.find("pds:axis_name", ns).text, int(a.find("pds:elements", ns).text))
                for a in array.findall("pds:Axis_Array", ns)]
        return dict(axes)
    except Exception as e:
        logging.error(f"Failed to parse XML {xml_path}: {e}")
        raise

def merge_orbit_files(orbit_dir, orbit, force=False):
    """Merge pre-filtered Parquet files for one orbit into a single file using Dask-Pandas."""
    output_path = orbit_dir / f"o_{orbit}_combined.parquet"
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
        # Load small files with Pandas
        geom_path = orbit_dir / f"o_{orbit}_geom.parquet"
        iono_path = orbit_dir / f"o_{orbit}_ionosphere.parquet"
        radar_path = orbit_dir / f"o_{orbit}_optim_no-wind_depth_resamp.parquet"
        mola_path = orbit_dir / f"o_{orbit}_MOLA_shaded_resamp.parquet"

        # Check if files exist
        for path in [geom_path, iono_path, radar_path, mola_path]:
            if not path.exists():
                raise FileNotFoundError(f"Required file not found: {path}")

        geom_df = pd.read_parquet(geom_path)
        iono_df = pd.read_parquet(iono_path)

        # Load pre-filtered radar and MOLA with Dask
        radar_ddf = dd.read_parquet(radar_path, blocksize="64MB")
        mola_ddf = dd.read_parquet(mola_path, blocksize="64MB")
        radar_rows = radar_ddf.shape[0].compute()
        radar_unique = radar_ddf[['Sample', 'Channel']].drop_duplicates().shape[0].compute()
        print(f"radar_ddf row count before merge: {radar_rows}")
        print(f"radar_ddf unique Sample/Channel pairs: {radar_unique}")
        logging.info(f"radar_ddf rows: {radar_rows}, unique Sample/Channel: {radar_unique}")
        log_memory()

        # Calculate Sample for geom_df dynamically
        num_samples = radar_ddf["Sample"].max().compute() + 1
        geom_df["Sample"] = ((geom_df["Frame"] - 1) * (num_samples / 978)).astype("int64")
        geom_df["Frame"] = geom_df["Frame"].astype("int64")
        geom_df["Sample"] = geom_df["Sample"].astype("int64")
       
            # Check for duplicates in Sample
        if geom_df["Sample"].duplicated().any():
            print(f"Warning: Duplicate Sample values found in geom_df: {geom_df[geom_df['Sample'].duplicated()]['Sample'].unique()}")
            # Drop duplicates, keeping the first occurrence
            geom_df = geom_df.drop_duplicates(subset=["Sample"], keep="first")
            
        geom_df = geom_df.set_index("Sample").reindex(range(num_samples)).reset_index()

        geom_df["Frame"] = geom_df["Frame"].interpolate(method="linear").ffill().bfill().astype("int64")
        geom_df["Latitude"] = geom_df["Latitude"].interpolate(method="linear").ffill().bfill()
        geom_df["Longitude"] = geom_df["Longitude"].interpolate(method="linear").ffill().bfill()
        geom_df["Altitude"] = geom_df["Altitude"].interpolate(method="linear").ffill().bfill()
        geom_df["Time"] = pd.to_datetime(geom_df["Time"].interpolate(method="linear").ffill().bfill(), unit='ns')
        print(f"geom_df dtypes: {geom_df.dtypes}")
        print(f"geom_df unique Samples: {len(geom_df['Sample'].unique())} (expected: {num_samples})")
        logging.info(f"geom_df unique Samples: {len(geom_df['Sample'].unique())} (expected: {num_samples})")
        log_memory()

        # Rename iono columns before converting to Dask
        iono_df = iono_df.rename(columns={
            "Channel 1": "Channel_1_Freq", "Channel 2": "Channel_2_Freq",
            "Delay 1": "Delay_1", "Delay 2": "Delay_2",
            "Channel 1 TEC1": "Channel_1_TEC1", "Channel 1 TEC2": "Channel_1_TEC2",
            "Channel 2 TEC1": "Channel_2_TEC1", "Channel 2 TEC2": "Channel_2_TEC2"
        })
        iono_df["Frame"] = iono_df["Frame"].astype("int64")
        print(f"iono_df dtypes: {iono_df.dtypes}")
        log_memory()

        # Convert to Dask
        geom_ddf = dd.from_pandas(geom_df, npartitions=8)
        iono_ddf = dd.from_pandas(iono_df, npartitions=8)

        # Debug: Confirm duplicate dropping is reached
        print(f"DEBUG: Processing radar_ddf for orbit {orbit}")
        # Check and drop duplicates in radar_ddf before merging with geom
        radar_ddf = radar_ddf.map_partitions(
            lambda df: df.drop_duplicates(subset=["Sample"], keep="first")
        )
        radar_rows_cleaned = radar_ddf.shape[0].compute()
        if radar_rows != radar_rows_cleaned:
            dropped = radar_rows - radar_rows_cleaned
            logging.error(f"Orbit {orbit}: Dropped {dropped} rows from radar_ddf due to duplicate Sample values")
            print(f"Orbit {orbit}: Dropped {dropped} rows from radar_ddf due to duplicate Sample values")
            logging.info(f"After dropping duplicates, radar_ddf rows: {radar_rows_cleaned}")

        # Merge radar with geom on Sample
        print(f"DEBUG: Merging radar_ddf with geom_ddf for orbit {orbit}")
        combined_df = radar_ddf.merge(
            geom_ddf[["Sample", "Frame", "Latitude", "Longitude", "Altitude", "Time"]],
            on="Sample", how="left"
        )
        combined_df["Frame"] = combined_df["Frame"].fillna(0).astype("int64")
        combined_df["Sample"] = combined_df["Sample"].fillna(0).astype("int64")
        print(f"combined_df dtypes after geom merge: {combined_df.dtypes}")
        rows_after_geom = combined_df.shape[0].compute()
        print(f"combined_df row count after geom merge: {rows_after_geom}")
        logging.info(f"combined_df rows after geom merge: {rows_after_geom}")
        log_memory()

        # Debug: Confirm iono duplicate dropping
        print(f"DEBUG: Processing iono_ddf for orbit {orbit}")
        # Check and drop duplicates in iono_ddf before merging
        iono_ddf = iono_ddf.map_partitions(
            lambda df: df.drop_duplicates(subset=["Frame"], keep="first")
        )
        iono_rows = iono_ddf.shape[0].compute()
        if iono_rows != len(iono_df):
            dropped = len(iono_df) - iono_rows
            logging.error(f"Orbit {orbit}: Dropped {dropped} rows from iono_ddf due to duplicate Frame values")
            print(f"Orbit {orbit}: Dropped {dropped} rows from iono_ddf due to duplicate Frame values")
            logging.info(f"After dropping duplicates, iono_ddf rows: {iono_rows}")

        # Merge with ionosphere on Frame
        print(f"DEBUG: Merging combined_df with iono_ddf for orbit {orbit}")
        combined_df = combined_df.merge(
            iono_ddf[["Frame", "Channel_1_Freq", "Channel_2_Freq", "Delay_1", "Delay_2",
                      "Channel_1_TEC1", "Channel_1_TEC2", "Channel_2_TEC1", "Channel_2_TEC2"]],
            on="Frame", how="left"
        )
        log_memory()

        # Select MOLA nadir line
        mola_xml_path = orbit_dir / f"o_{orbit}_MOLA_shaded_resamp.xml"
        mola_dims = get_img_dims(mola_xml_path, "Array_2D_Image")
        nadir_line = mola_dims["Line"] // 2
        mola_nadir = mola_ddf[mola_ddf["Line"] == nadir_line][["Sample", "MOLA_Elevation"]]

        # Debug: Confirm MOLA duplicate dropping
        print(f"DEBUG: Processing mola_nadir for orbit {orbit}")
        # Check and drop duplicates in mola_nadir before merging
        mola_nadir = mola_nadir.map_partitions(
            lambda df: df.drop_duplicates(subset=["Sample"], keep="first")
        )
        mola_rows = mola_nadir.shape[0].compute()
        mola_rows_original = mola_ddf[mola_ddf["Line"] == nadir_line].shape[0].compute()
        if mola_rows != mola_rows_original:
            dropped = mola_rows_original - mola_rows
            logging.error(f"Orbit {orbit}: Dropped {dropped} rows from mola_nadir due to duplicate Sample values")
            print(f"Orbit {orbit}: Dropped {dropped} rows from mola_nadir due to duplicate Sample values")
            logging.info(f"After dropping duplicates, mola_nadir rows: {mola_rows}")

        # Merge with MOLA
        print(f"DEBUG: Merging combined_df with mola_nadir for orbit {orbit}")
        combined_df = combined_df.merge(mola_nadir, on="Sample", how="left")
        combined_df["Sample"] = combined_df["Sample"].fillna(0).astype("int64")
        log_memory()

        # Add computed columns
        combined_df["Orbit"] = orbit
        combined_df["Depth_Below_Ground_m"] = combined_df["Line"] * 26.767
        combined_df["Above_Below_Ellipsoid_m"] = combined_df["MOLA_Elevation"] - combined_df["Depth_Below_Ground_m"]
        combined_df["Frame"] = combined_df["Frame"].astype("int64")
        combined_df["UniqueKey"] = (combined_df["Orbit"] + "_" + combined_df["Frame"].astype(str) + "_" +
                                   combined_df["Sample"].astype(str) + "_" + combined_df["Line"].astype(str) + "_" +
                                   combined_df["Channel"].astype(str))
        log_memory()

        # Final columns (include Channel)
        final_cols = ["UniqueKey", "Orbit", "Frame", "Latitude", "Longitude", "Altitude", "Depth_Below_Ground_m",
                      "Raw_Power", "Channel", "Channel_1_Freq", "Channel_2_Freq", "Delay_1", "Delay_2", "Channel_1_TEC1", "Channel_1_TEC2",
                      "Channel_2_TEC1", "Channel_2_TEC2", "Time", "Sample", "Line", "MOLA_Elevation", "Above_Below_Ellipsoid_m"]
        combined_df = combined_df[final_cols]

        # Compute to cuDF and write single file
        print(f"DEBUG: Computing and writing combined_df for orbit {orbit}")
        combined_df = combined_df.compute()  # To Pandas
        combined_df = cudf.from_pandas(combined_df)  # To cuDF
        combined_df.to_parquet(output_path, index=False)  # Single file
        print(f"Merged orbit {orbit} to {output_path}, shape: {combined_df.shape}")
        logging.info(f"Merged orbit {orbit} to {output_path}, shape: {combined_df.shape}")
        log_memory()
        return output_path
    except FileNotFoundError as e:
        print(f"File not found error for orbit {orbit}: {e}")
        logging.error(f"File not found error for orbit {orbit}: {e}")
        return None
    except Exception as e:
        print(f"Error merging orbit {orbit}: {e}")
        logging.error(f"Error merging orbit {orbit}: {e}")
        return None

def main(force=False):
    print("This is the right script")  # Your added confirmation
    # Use the directory of the script as the root, not cwd
    script_dir = Path(__file__).parent
    print(f"Scanning directory and subdirectories: {script_dir}")
    logging.info(f"Scanning directory and subdirectories: {script_dir}")

    # Find all o_*_geom.parquet files recursively starting from script_dir
    orbits = set()
    for geom_file in script_dir.glob("**/o_*_geom.parquet"):
        parts = geom_file.name.split('_')
        if len(parts) < 2 or not parts[1].isdigit():
            continue
        orbit = parts[1][:5]
        orbits.add((orbit, geom_file.parent))

    if not orbits:
        print("No orbit Parquet files found in directory or subdirectories.")
        logging.warning("No orbit Parquet files found in directory or subdirectories.")
        return

    failed_orbits = []
    for orbit, orbit_dir in sorted(orbits):
        try:
            result = merge_orbit_files(orbit_dir, orbit, force=force)
            if result is None:
                failed_orbits.append(orbit)
            else:
                print(f"Successfully processed orbit {orbit}")
                logging.info(f"Successfully processed orbit {orbit}")
        except Exception as e:
            print(f"Unexpected error processing orbit {orbit}: {e}")
            logging.error(f"Unexpected error processing orbit {orbit}: {e}")
            failed_orbits.append(orbit)

    if failed_orbits:
        print(f"Failed orbits: {failed_orbits}")
        logging.info(f"Failed orbits: {failed_orbits}")
    else:
        print("All orbits processed successfully!")
        logging.info("All orbits processed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge orbit Parquet files into a single file with Dask-Pandas.")
    parser.add_argument("--force", action="store_true", help="Force regeneration")
    args = parser.parse_args()
    main(args.force)