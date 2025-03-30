import os
import shutil
import argparse
import logging
import logging.handlers
from datetime import datetime
import numpy as np
import pandas as pd
import cudf  # For GPU acceleration
import dask.dataframe as dd
from dask.distributed import Client
from pathlib import Path
import psutil
from xml.etree import ElementTree as ET

# Set up logging with a single handler
logger = logging.getLogger()
if not logger.hasHandlers():
    # File handler
    file_handler = logging.handlers.TimedRotatingFileHandler(
        filename=f"merge_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
        when="midnight",
        interval=1,
        backupCount=7
    )
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Stream handler for console output
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    
    logger.setLevel(logging.INFO)

# Redirect Dask logging to our logger
logging.getLogger('distributed').setLevel(logging.WARNING)
logging.getLogger('distributed').handlers = logger.handlers

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

def compute_ground_depth(radar_gdf, depth_per_pixel=26.767, max_depth_m=5000):
    """Detect ground line and compute depth using cuDF for GPU acceleration."""
    print(f"radar_gdf rows before depth calc: {len(radar_gdf)}")
    logging.info(f"radar_gdf rows before depth calc: {len(radar_gdf)}")
    radar_gdf["Computed_Power_dB"] = (radar_gdf["Raw_Power"] * 0.4) + 40
    print(f"Raw_Power range: {radar_gdf['Raw_Power'].min()} to {radar_gdf['Raw_Power'].max()}")
    print(f"Computed_Power_dB range: {radar_gdf['Computed_Power_dB'].min()} to {radar_gdf['Computed_Power_dB'].max()}")
    logging.info(f"Raw_Power range: {radar_gdf['Raw_Power'].min()} to {radar_gdf['Raw_Power'].max()}")
    logging.info(f"Computed_Power_dB range: {radar_gdf['Computed_Power_dB'].min()} to {radar_gdf['Computed_Power_dB'].max()}")
    
    # Log available channels
    channels = radar_gdf["Channel"].unique().to_pandas().values
    print(f"Available channels: {channels}")
    logging.info(f"Available channels: {channels}")
    
    # Get all unique samples upfront
    all_samples = radar_gdf["Sample"].unique()
    print(f"Total unique samples: {len(all_samples)}")
    logging.info(f"Total unique samples: {len(all_samples)}")
    
    # Group by Sample to find the minimum channel per sample
    min_channels = radar_gdf.groupby("Sample")["Channel"].min().reset_index().rename(columns={"Channel": "Min_Channel"})
    
    # Debug: Check min_channels
    print(f"min_channels rows: {len(min_channels)}")
    logging.info(f"min_channels rows: {len(min_channels)}")
    if min_channels.empty:
        logging.warning("min_channels is empty, defaulting to first channel")
        min_channels = cudf.DataFrame({"Sample": all_samples, "Min_Channel": channels[0]})
    
    # Ensure min_channels covers all samples
    min_channels_all = cudf.DataFrame({"Sample": all_samples})
    min_channels = min_channels_all.merge(min_channels, on="Sample", how="left")
    min_channels["Min_Channel"] = min_channels["Min_Channel"].fillna(channels[0])  # Default to first channel if missing
    
    # Debug: Check Min_Channel values
    print(f"Min_Channel range: {min_channels['Min_Channel'].min()} to {min_channels['Min_Channel'].max()}")
    logging.info(f"Min_Channel range: {min_channels['Min_Channel'].min()} to {min_channels['Min_Channel'].max()}")
    print(f"Min_Channel NaN count: {min_channels['Min_Channel'].isna().sum()}")
    logging.info(f"Min_Channel NaN count: {min_channels['Min_Channel'].isna().sum()}")
    
    # Filter to keep only the minimum channel for each sample
    radar_gdf = radar_gdf.merge(min_channels, on="Sample", how="left")
    channel_data = radar_gdf[radar_gdf["Channel"] == radar_gdf["Min_Channel"]]
    
    # Debug: Check channel_data
    print(f"channel_data rows: {len(channel_data)}")
    logging.info(f"channel_data rows: {len(channel_data)}")
    if channel_data.empty:
        logging.warning("No data for minimum channels, defaulting all ground lines to 0")
        ground_lines = cudf.Series(index=all_samples, data=0, dtype="int64")
    else:
        # Debug: Check Sample values in channel_data
        print(f"channel_data Sample NaN count: {channel_data['Sample'].isna().sum()}")
        logging.info(f"channel_data Sample NaN count: {channel_data['Sample'].isna().sum()}")
        print(f"channel_data Sample dtype: {channel_data['Sample'].dtype}")
        logging.info(f"channel_data Sample dtype: {channel_data['Sample'].dtype}")
        
        # Detect ground lines using cuDF vectorized operations
        print("Detecting ground lines with cuDF...")
        # Sort by Line to ensure we get the first occurrence
        channel_data = channel_data.sort_values(["Sample", "Line"])
        
        # Find first Line where Computed_Power_dB > 45
        above_45 = channel_data[channel_data["Computed_Power_dB"] > 45]
        ground_lines_45 = above_45.groupby("Sample").head(1)[["Sample", "Line"]].rename(columns={"Line": "Ground_Line"})
        
        # Debug: Check ground_lines_45 and remaining_data
        print(f"ground_lines_45 rows: {len(ground_lines_45)}")
        logging.info(f"ground_lines_45 rows: {len(ground_lines_45)}")
        samples_45 = ground_lines_45["Sample"]
        remaining_data = channel_data[~channel_data["Sample"].isin(samples_45)]
        print(f"remaining_data rows: {len(remaining_data)}")
        logging.info(f"remaining_data rows: {len(remaining_data)}")
        
        # For samples without > 45, find first Line where > 40
        above_40 = remaining_data[remaining_data["Computed_Power_dB"] > 40]
        print(f"above_40 rows: {len(above_40)}")
        logging.info(f"above_40 rows: {len(above_40)}")
        if above_40.empty:
            ground_lines_40 = cudf.DataFrame(columns=["Sample", "Ground_Line"], dtype="int64")
        else:
            ground_lines_40 = above_40.groupby("Sample").head(1)[["Sample", "Line"]].rename(columns={"Line": "Ground_Line"})
        
        # Combine and fill remaining samples with 0
        ground_lines_df = cudf.concat([ground_lines_45, ground_lines_40], ignore_index=True)
        print(f"ground_lines_df rows: {len(ground_lines_df)}")
        logging.info(f"ground_lines_df rows: {len(ground_lines_df)}")
        print(f"ground_lines_df columns: {ground_lines_df.columns}")
        logging.info(f"ground_lines_df columns: {ground_lines_df.columns}")
        if ground_lines_df.empty:
            logging.warning("No ground lines detected, defaulting all to 0")
            ground_lines = cudf.Series(index=all_samples, data=0, dtype="int64")
        else:
            # Ensure 'Sample' column exists before setting index
            if "Sample" not in ground_lines_df.columns:
                logging.error("Sample column missing in ground_lines_df")
                raise KeyError("Sample")
            ground_lines = ground_lines_df.set_index("Sample")["Ground_Line"]
            ground_lines = cudf.Series(index=all_samples, data=0, dtype="int64").add(ground_lines, fill_value=0).astype("int64")
    
    # Log ground line stats
    ground_line_values = ground_lines.to_pandas().values
    print(f"Ground line range: {min(ground_line_values)} to {max(ground_line_values)}")
    logging.info(f"Ground line range: {min(ground_line_values)} to {max(ground_line_values)}")
    
    # Map ground lines to all rows
    radar_gdf = radar_gdf.merge(ground_lines.rename("Ground_Line"), on="Sample", how="left")
    radar_gdf["Depth_Below_Ground_m"] = (radar_gdf["Line"] - radar_gdf["Ground_Line"]) * depth_per_pixel * -1
    
    # Filter to keep only subsurface (0 to -max_depth_m)
    radar_gdf = radar_gdf[(radar_gdf["Depth_Below_Ground_m"] >= -max_depth_m) & (radar_gdf["Depth_Below_Ground_m"] <= 0)]
    radar_gdf = radar_gdf.drop(columns=["Computed_Power_dB", "Min_Channel", "Ground_Line"])
    return radar_gdf

def merge_orbit_files(orbit_dir, orbit, force=False):
    """Merge pre-filtered Parquet files for one orbit into a single file using cuDF."""
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
        geom_path = orbit_dir / f"o_{orbit}_geom.parquet"
        iono_path = orbit_dir / f"o_{orbit}_ionosphere.parquet"
        radar_path = orbit_dir / f"o_{orbit}_optim_no-wind_depth_resamp.parquet"
        mola_path = orbit_dir / f"o_{orbit}_MOLA_shaded_resamp.parquet"

        for path in [geom_path, iono_path, radar_path, mola_path]:
            if not path.exists():
                raise FileNotFoundError(f"Required file not found: {path}")

        # Load all data directly into cuDF
        geom_gdf = cudf.read_parquet(geom_path)
        iono_gdf = cudf.read_parquet(iono_path)
        radar_gdf = cudf.read_parquet(radar_path)
        mola_gdf = cudf.read_parquet(mola_path)

        # Debug: Check geom_gdf columns after loading
        print(f"geom_gdf columns after loading: {geom_gdf.columns}")
        logging.info(f"geom_gdf columns after loading: {geom_gdf.columns}")

        # Compute depth on GPU
        radar_gdf = compute_ground_depth(radar_gdf)
        radar_rows = len(radar_gdf)
        radar_unique = radar_gdf[['Sample', 'Channel']].drop_duplicates().shape[0]
        print(f"radar_gdf row count after depth calc: {radar_rows}")
        print(f"radar_gdf unique Sample/Channel pairs: {radar_unique}")
        logging.info(f"radar_gdf rows: {radar_rows}, unique Sample/Channel: {radar_unique}")
        log_memory()

        num_samples = radar_gdf["Sample"].max() + 1
        geom_gdf["Sample"] = ((geom_gdf["Frame"] - 1) * (num_samples / 978)).astype("int64")
        geom_gdf["Frame"] = geom_gdf["Frame"].astype("int64")
        geom_gdf["Sample"] = geom_gdf["Sample"].astype("int64")

        # Debug: Check geom_gdf columns after adding Sample
        print(f"geom_gdf columns after adding Sample: {geom_gdf.columns}")
        logging.info(f"geom_gdf columns after adding Sample: {geom_gdf.columns}")

        # Drop duplicates in geom_gdf
        if "Sample" not in geom_gdf.columns:
            logging.error("Sample column missing in geom_gdf after adding")
            raise KeyError("Sample")
        geom_gdf = geom_gdf.drop_duplicates(subset=["Sample"], keep="first")

        # Debug: Check geom_gdf columns after drop_duplicates
        print(f"geom_gdf columns after drop_duplicates: {geom_gdf.columns}")
        logging.info(f"geom_gdf columns after drop_duplicates: {geom_gdf.columns}")

        # Reindex and interpolate
        if "Sample" not in geom_gdf.columns:
            logging.error("Sample column missing in geom_gdf before reindex")
            raise KeyError("Sample")
        # Explicitly set index name to ensure reset_index creates 'Sample' column
        geom_gdf = geom_gdf.set_index("Sample")
        print(f"geom_gdf index name after set_index: {geom_gdf.index.name}")
        logging.info(f"geom_gdf index name after set_index: {geom_gdf.index.name}")
        geom_gdf = geom_gdf.reindex(cudf.Index(range(num_samples)))
        print(f"geom_gdf index name after reindex: {geom_gdf.index.name}")
        logging.info(f"geom_gdf index name after reindex: {geom_gdf.index.name}")
        # Ensure the index name is preserved
        geom_gdf.index.name = "Sample"
        geom_gdf = geom_gdf.reset_index()
#        geom_gdf["Frame"] = geom_gdf["Frame"].interpolate(method="linear").ffill().bfill().astype("int64")
#        geom_gdf["Latitude"] = geom_gdf["Latitude"].interpolate(method="linear").ffill().bfill()
#        geom_gdf["Longitude"] = geom_gdf["Longitude"].interpolate(method="linear").ffill().bfill()
#        geom_gdf["Altitude"] = geom_gdf["Altitude"].interpolate(method="linear").ffill().bfill()
#        geom_gdf["Time"] = pd.to_datetime(geom_gdf["Time"].interpolate(method="linear").ffill().bfill(), unit='ns')

        # Debug: Check geom_gdf Time column before interpolation
        print(f"geom_gdf Time dtype before interpolation: {geom_gdf['Time'].dtype}")
        logging.info(f"geom_gdf Time dtype before interpolation: {geom_gdf['Time'].dtype}")
        print(f"geom_gdf Time sample values before interpolation: {geom_gdf['Time'].head(5).to_pandas()}")
        logging.info(f"geom_gdf Time sample values before interpolation: {geom_gdf['Time'].head(5).to_pandas()}")

        # Ensure Time is in datetime format
        if not str(geom_gdf["Time"].dtype).startswith("datetime"):
            geom_gdf["Time"] = cudf.to_datetime(geom_gdf["Time"])
            print(f"Converted geom_gdf Time to datetime: {geom_gdf['Time'].dtype}")
            logging.info(f"Converted geom_gdf Time to datetime: {geom_gdf['Time'].dtype}")

        # Interpolate and fill
        geom_gdf["Frame"] = geom_gdf["Frame"].interpolate(method="linear").ffill().bfill().astype("int64")
        geom_gdf["Latitude"] = geom_gdf["Latitude"].interpolate(method="linear").ffill().bfill()
        geom_gdf["Longitude"] = geom_gdf["Longitude"].interpolate(method="linear").ffill().bfill()
        geom_gdf["Altitude"] = geom_gdf["Altitude"].interpolate(method="linear").ffill().bfill()
        # Interpolate Time as numeric (nanoseconds), then convert back to datetime
        time_ns = geom_gdf["Time"].astype("int64")  # Convert to nanoseconds
        time_ns = time_ns.interpolate(method="linear").ffill().bfill()
        geom_gdf["Time"] = cudf.to_datetime(time_ns)
        
        # Debug: Check geom_gdf Time column after interpolation
        print(f"geom_gdf Time dtype after interpolation: {geom_gdf['Time'].dtype}")
        logging.info(f"geom_gdf Time dtype after interpolation: {geom_gdf['Time'].dtype}")
        print(f"geom_gdf Time sample values after interpolation: {geom_gdf['Time'].head(5).to_pandas()}")
        logging.info(f"geom_gdf Time sample values after interpolation: {geom_gdf['Time'].head(5).to_pandas()}")



        # Debug: Check geom_gdf columns after reindex
        print(f"geom_gdf columns after reindex: {geom_gdf.columns}")
        logging.info(f"geom_gdf columns after reindex: {geom_gdf.columns}")

        if "Sample" not in geom_gdf.columns:
            logging.error("Sample column missing in geom_gdf after reindex")
            raise KeyError("Sample")
        print(f"geom_gdf dtypes: {geom_gdf.dtypes}")
        print(f"geom_gdf unique Samples: {len(geom_gdf['Sample'].unique())} (expected: {num_samples})")
        logging.info(f"geom_gdf unique Samples: {len(geom_gdf['Sample'].unique())} (expected: {num_samples})")
        log_memory()

        iono_gdf = iono_gdf.rename(columns={
            "Channel 1": "Channel_1_Freq", "Channel 2": "Channel_2_Freq",
            "Delay 1": "Delay_1", "Delay 2": "Delay_2",
            "Channel 1 TEC1": "Channel_1_TEC1", "Channel 1 TEC2": "Channel_1_TEC2",
            "Channel 2 TEC1": "Channel_2_TEC1", "Channel 2 TEC2": "Channel_2_TEC2"
        })
        iono_gdf["Frame"] = iono_gdf["Frame"].astype("int64")
        print(f"iono_gdf dtypes: {iono_gdf.dtypes}")
        log_memory()

        print(f"DEBUG: Merging radar_gdf with geom_gdf for orbit {orbit}")
        combined_gdf = radar_gdf.merge(
            geom_gdf[["Sample", "Frame", "Latitude", "Longitude", "Altitude", "Time"]],
            on="Sample", how="left"
        )
        combined_gdf["Frame"] = combined_gdf["Frame"].fillna(0).astype("int64")
        combined_gdf["Sample"] = combined_gdf["Sample"].fillna(0).astype("int64")
        print(f"combined_gdf dtypes after geom merge: {combined_gdf.dtypes}")
        rows_after_geom = len(combined_gdf)
        print(f"combined_gdf row count after geom merge: {rows_after_geom}")
        logging.info(f"combined_gdf rows after geom merge: {rows_after_geom}")
        log_memory()

        print(f"DEBUG: Merging combined_gdf with iono_gdf for orbit {orbit}")
        combined_gdf = combined_gdf.merge(
            iono_gdf[["Frame", "Channel_1_Freq", "Channel_2_Freq", "Delay_1", "Delay_2",
                      "Channel_1_TEC1", "Channel_1_TEC2", "Channel_2_TEC1", "Channel_2_TEC2"]],
            on="Frame", how="left"
        )
        log_memory()

        mola_xml_path = orbit_dir / f"o_{orbit}_MOLA_shaded_resamp.xml"
        mola_dims = get_img_dims(mola_xml_path, "Array_2D_Image")
        nadir_line = mola_dims["Line"] // 2
        mola_nadir = mola_gdf[mola_gdf["Line"] == nadir_line][["Sample", "MOLA_Elevation"]]

        print(f"DEBUG: Merging combined_gdf with mola_nadir for orbit {orbit}")
        combined_gdf = combined_gdf.merge(mola_nadir, on="Sample", how="left")
        combined_gdf["Sample"] = combined_gdf["Sample"].fillna(0).astype("int64")
        log_memory()

        combined_gdf["Orbit"] = orbit
        combined_gdf["Above_Below_Ellipsoid_m"] = combined_gdf["MOLA_Elevation"] - combined_gdf["Depth_Below_Ground_m"]
        combined_gdf["Frame"] = combined_gdf["Frame"].astype("int64")
        combined_gdf["UniqueKey"] = (combined_gdf["Orbit"] + "_" + combined_gdf["Frame"].astype(str) + "_" +
                                   combined_gdf["Sample"].astype(str) + "_" + combined_gdf["Line"].astype(str) + "_" +
                                   combined_gdf["Channel"].astype(str))
        log_memory()

        final_cols = ["UniqueKey", "Orbit", "Frame", "Latitude", "Longitude", "Altitude", "Depth_Below_Ground_m",
                      "Raw_Power", "Channel", "Channel_1_Freq", "Channel_2_Freq", "Delay_1", "Delay_2", "Channel_1_TEC1", "Channel_1_TEC2",
                      "Channel_2_TEC1", "Channel_2_TEC2", "Time", "Sample", "Line", "MOLA_Elevation", "Above_Below_Ellipsoid_m"]
        combined_gdf = combined_gdf[final_cols]

        print(f"DEBUG: Writing combined_gdf for orbit {orbit}")
        combined_gdf.to_parquet(output_path, index=False)
        print(f"Merged orbit {orbit} to {output_path}, shape: {combined_gdf.shape}")
        logging.info(f"Merged orbit {orbit} to {output_path}, shape: {combined_gdf.shape}")
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
    finally:
        # Ensure log file is written
        for handler in logger.handlers:
            handler.flush()

def main(force=False):
    # Use context manager for clean Dask client lifecycle
    with Client(
        n_workers=max(1, os.cpu_count() // 2),
        threads_per_worker=1,
        processes=False,  # Inproc workers
        heartbeat_interval="5s"  # Slower heartbeat to reduce contention
    ) as client:
        print(f"Dask client started with {max(1, os.cpu_count() // 2)} workers")
        logging.info(f"Dask client started with {max(1, os.cpu_count() // 2)} workers")

        script_dir = Path(__file__).parent
        print(f"Scanning directory and subdirectories: {script_dir}")
        logging.info(f"Scanning directory and subdirectories: {script_dir}")

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

    # Flush logs after client context exits
    for handler in logger.handlers:
        handler.flush()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge orbit Parquet files into a single file with Dask-Pandas.")
    parser.add_argument("--force", action="store_true", help="Force regeneration")
    args = parser.parse_args()
    main(args.force)