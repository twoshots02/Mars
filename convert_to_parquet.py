import os
import shutil
import argparse
import logging
from datetime import datetime
import numpy as np
import pandas as pd
import cudf  # RAPIDS GPU DataFrame
from pathlib import Path
from xml.etree import ElementTree as ET

# Logging setup
logging.basicConfig(
    filename=f"convert_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def get_xml_fields(xml_path, tag="Field_Delimited"):
    """Extract column names from XML for TAB files."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    ns = {"pds": "http://pds.nasa.gov/pds4/pds/v1"}
    fields = [f.find("pds:name", ns).text for f in root.findall(f".//pds:{tag}", ns)]
    if not fields:
        raise ValueError(f"No {tag} elements found in {xml_path}")
    return fields

def get_img_dims(xml_path, array_type):
    """Extract array dimensions and data type from XML for IMG files."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    ns = {"pds": "http://pds.nasa.gov/pds4/pds/v1"}
    array = root.find(f".//pds:{array_type}", ns)
    if array is None:
        raise ValueError(f"No {array_type} found in {xml_path}")
    axes = [(a.find("pds:axis_name", ns).text, int(a.find("pds:elements", ns).text))
            for a in array.findall("pds:Axis_Array", ns)]
    dtype_elem = array.find("pds:Element_Array/pds:data_type", ns)
    if dtype_elem is None:
        raise ValueError(f"No data_type found in {xml_path}")
    dtype = np.uint16 if "LSB2" in dtype_elem.text else np.uint8
    scale_elem = array.find("pds:Element_Array/pds:scaling_factor", ns)
    offset_elem = array.find("pds:Element_Array/pds:value_offset", ns)
    scale = float(scale_elem.text) if scale_elem is not None else 1.0
    offset = float(offset_elem.text) if offset_elem is not None else 0.0
    return dict(axes), dtype, scale, offset

def convert_tab_to_parquet(tab_path, xml_path, output_path, force=False, keep_cols=None):
    """Convert TAB (CSV) to Parquet, keeping only specified columns."""
    if not force and output_path.exists() and output_path.stat().st_size > 0:
        print(f"Skipping existing Parquet: {output_path}")
        logging.info(f"Skipping existing Parquet: {output_path}")
        return output_path
    if force and output_path.exists():
        os.remove(output_path)
        print(f"Force deleted: {output_path}")
        logging.info(f"Force deleted: {output_path}")

    try:
        cols = get_xml_fields(xml_path)
        df = cudf.read_csv(tab_path, delimiter=",", names=cols, header=None)
        df_pd = df.to_pandas()
        df_pd["Time"] = pd.to_datetime(df_pd["Time"], utc=True, errors="coerce").dt.tz_localize(None)
        df_pd = df_pd[df_pd["Time"].notna()]
        df = cudf.from_pandas(df_pd)
        if keep_cols:
            df = df[[c for c in keep_cols if c in df.columns]]
        df.to_parquet(output_path, index=False)
        print(f"Converted {tab_path} to {output_path}, shape: {df.shape}")
        logging.info(f"Converted {tab_path} to {output_path}, shape: {df.shape}")
        return output_path
    except Exception as e:
        print(f"Error converting {tab_path}: {e}")
        logging.error(f"Error converting {tab_path}: {e}")
        raise

def convert_img_to_parquet(img_path, xml_path, output_path, force=False):
    """Convert IMG (binary array) to Parquet, filtering radar to subsurface."""
    if not force and output_path.exists() and output_path.stat().st_size > 0:
        print(f"Skipping existing Parquet: {output_path}")
        logging.info(f"Skipping existing Parquet: {output_path}")
        return output_path
    if force and output_path.exists():
        os.remove(output_path)
        print(f"Force deleted: {output_path}")
        logging.info(f"Force deleted: {output_path}")

    try:
        dims, dtype, scale, offset = get_img_dims(xml_path, "Array_2D_Image" if "MOLA" in img_path.name else "Array_3D_Image")
        data = np.fromfile(img_path, dtype=dtype)
        expected_size = np.prod([v for _, v in dims.items()])
        if data.size != expected_size:
            print(f"{img_path}: Expected {expected_size} elements, got {data.size}—truncating")
            logging.warning(f"{img_path}: Expected {expected_size} elements, got {data.size}")
            data = data[:expected_size]

        if "MOLA" in img_path.name:
            data = data.reshape(dims["Line"], dims["Sample"])
            df = cudf.DataFrame({
                "Line": np.repeat(np.arange(dims["Line"]), dims["Sample"]),
                "Sample": np.tile(np.arange(dims["Sample"]), dims["Line"]),
                "MOLA_Elevation": data.ravel()
            })
        else:  # Radar
            data = data.reshape(dims["Channel"], dims["Line"], dims["Sample"])
            power = data * scale + offset  # dB scaling
            df = cudf.DataFrame({
                "Channel": np.repeat([1, 2], dims["Line"] * dims["Sample"]),
                "Line": np.tile(np.repeat(np.arange(dims["Line"]), dims["Sample"]), 2),
                "Sample": np.tile(np.arange(dims["Sample"]), 2 * dims["Line"]),
                "Raw_Power": power.ravel()
            })
            # Filter to subsurface
            threshold_dB = 45
            max_depth_m = 5000
            depth_per_pixel = 26.767
            max_depth_idx = int(max_depth_m / depth_per_pixel)  # ~187 lines
            ground_levels = df[df["Raw_Power"] > threshold_dB].groupby(["Channel", "Sample"])["Line"].min().reset_index(name="Ground_Line")
            df = df.merge(ground_levels, on=["Channel", "Sample"], how="left")
            df["Ground_Line"] = df["Ground_Line"].fillna(0)  # Default to 0 if no threshold met
            df = df[
                (df["Line"] >= df["Ground_Line"]) & 
                (df["Line"] <= df["Ground_Line"] + max_depth_idx)
            ].drop(columns=["Ground_Line"])

        df.to_parquet(output_path, index=False)
        print(f"Converted {img_path} to {output_path}, shape: {df.shape}")
        logging.info(f"Converted {img_path} to {output_path}, shape: {df.shape}")
        return output_path
    except Exception as e:
        print(f"Error converting {img_path}: {e}")
        logging.error(f"Error converting {img_path}: {e}")
        raise

def find_xml_file(orbit_dir, base_name):
    """Find an XML file matching the base name, allowing for variations."""
    # Look for exact match first
    exact_path = orbit_dir / f"{base_name}.xml"
    if exact_path.exists():
        return exact_path
    # Look for variations (e.g., o_01883_optim_no-wind_depth_resamp - Copy.xml)
    xml_files = list(orbit_dir.glob(f"{base_name}*.xml"))
    # Filter out files with unwanted suffixes like :Zone.Identifier
    xml_files = [f for f in xml_files if not str(f).endswith(":Zone.Identifier")]
    if xml_files:
        # Sort by name length to prefer the shortest (most likely the correct one)
        xml_files.sort(key=lambda x: len(x.name))
        return xml_files[0]
    return None

def main(force=False):
    root = Path.cwd()
    print(f"Scanning current directory and subdirectories: {root}")
    logging.info(f"Scanning current directory and subdirectories: {root}")

    # If --force, delete all existing o_*_*.parquet files in subdirectories
    if force:
        parquet_files = list(root.glob("**/o_*_*.parquet"))
        for parquet_file in parquet_files:
            if parquet_file.is_file():
                os.remove(parquet_file)
                print(f"Force deleted existing Parquet: {parquet_file}")
                logging.info(f"Force deleted existing Parquet: {parquet_file}")

    files_by_orbit = {}
    # Recursively search for o_* raw files in all subdirectories
    for file in root.glob("**/o_*"):
        if not file.is_file():
            continue  # Skip directories
        # Extract orbit number (e.g., 01867 from o_01867_geom.tab)
        parts = file.name.split('_')
        if len(parts) < 2 or not parts[1].isdigit():
            print(f"Skipping file with invalid orbit format: {file}")
            logging.warning(f"Skipping file with invalid orbit format: {file}")
            continue
        orbit = parts[1][:5]  # Extract 5-digit orbit number
        if orbit not in files_by_orbit:
            files_by_orbit[orbit] = {
                "geom": None, "iono": None, "mola": None, "radar": None,
                "geom_xml": None, "iono_xml": None, "mola_xml": None, "radar_xml": None
            }
        # Only assign raw files, not XMLs (we'll find XMLs later)
        if file.name.endswith("_geom.tab"):
            files_by_orbit[orbit]["geom"] = file
        elif file.name.endswith("_ionosphere.tab"):
            files_by_orbit[orbit]["iono"] = file
        elif file.name.endswith("_MOLA_shaded_resamp.img"):
            files_by_orbit[orbit]["mola"] = file
        elif file.name.endswith("_optim_no-wind_depth_resamp.img"):
            files_by_orbit[orbit]["radar"] = file
        else:
            print(f"Ignoring unrecognized file: {file}")
            logging.info(f"Ignoring unrecognized file: {file}")

    if not files_by_orbit:
        print("No 'o_*' files found in current directory or subdirectories.")
        logging.warning("No 'o_*' files found in current directory or subdirectories.")
        return

    for orbit, files in files_by_orbit.items():
        print(f"Processing orbit: {orbit}")
        logging.info(f"Processing orbit: {orbit}")
        # Check if there are any raw files for this orbit
        raw_files = [files["geom"], files["iono"], files["mola"], files["radar"]]
        if not any(raw_files):
            print(f"Skipping orbit {orbit}: No raw files found")
            logging.warning(f"Skipping orbit {orbit}: No raw files found")
            continue

        # Determine the directory for this orbit
        orbit_dir = files["geom"].parent if files["geom"] else files["iono"].parent if files["iono"] else files["mola"].parent if files["mola"] else files["radar"].parent
        # Find XMLs for each raw file
        if files["geom"]:
            files["geom_xml"] = find_xml_file(orbit_dir, f"o_{orbit}_geom")
        if files["iono"]:
            files["iono_xml"] = find_xml_file(orbit_dir, f"o_{orbit}_ionosphere")
        if files["mola"]:
            files["mola_xml"] = find_xml_file(orbit_dir, f"o_{orbit}_MOLA_shaded_resamp")
        if files["radar"]:
            files["radar_xml"] = find_xml_file(orbit_dir, f"o_{orbit}_optim_no-wind_depth_resamp")

        # Check if all required files and XMLs are present
        required_keys = ["geom", "iono", "mola", "radar", "geom_xml", "iono_xml", "mola_xml", "radar_xml"]
        missing_keys = [key for key in required_keys if not files[key]]
        if missing_keys:
            print(f"Skipping orbit {orbit}: Missing required files or XMLs: {missing_keys}")
            logging.warning(f"Skipping orbit {orbit}: Missing required files or XMLs: {missing_keys}")
            continue

        geom_cols = ["Frame", "Latitude", "Longitude", "Altitude", "Time"]
        iono_cols = ["Frame", "Time", "Channel 1", "Channel 2", "Delay 1", "Delay 2",
                     "Channel 1 TEC1", "Channel 1 TEC2", "Channel 2 TEC1", "Channel 2 TEC2"]
        outputs = {}
        for key, path in files.items():
            if key in ["geom", "iono", "mola", "radar"]:
                xml_key = key + "_xml"
                out_path = path.with_suffix(".parquet")
                if "tab" in path.name:
                    cols = geom_cols if "geom" in key else iono_cols
                    outputs[key] = convert_tab_to_parquet(path, files[xml_key], out_path, force, cols)
                elif "img" in path.name:
                    outputs[key] = convert_img_to_parquet(path, files[xml_key], out_path, force)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Mars data files to Parquet from current directory and subdirectories.")
    parser.add_argument("--force", action="store_true", help="Force regeneration")
    args = parser.parse_args()
    main(args.force)