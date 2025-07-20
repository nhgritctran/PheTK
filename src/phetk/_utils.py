from concurrent.futures import ThreadPoolExecutor, as_completed
from google.cloud import bigquery
from itertools import combinations
from tqdm import tqdm

import argparse
import csv
import os
import polars as pl
import psutil
import pyarrow as pa
import sys
import time


def str_to_bool(v) -> bool:
    """
    Convert string to boolean for argparse.
    
    :param v: Input value to convert to boolean
    :type v: str | bool
    :return: Boolean value
    :rtype: bool
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def to_polars(df) -> pl.DataFrame:
    """
    Check and convert pandas dataframe object to polars dataframe, if applicable.
    
    :param df: Input dataframe object (pandas DataFrame or polars DataFrame)
    :type df: Any
    :return: Polars DataFrame
    :rtype: pl.DataFrame
    """
    if not isinstance(df, pl.DataFrame):
        return pl.from_pandas(df)
    else:
        return df


def spark_to_polars(spark_df) -> pl.DataFrame:
    """
    Convert Spark DataFrame to Polars DataFrame.
    
    :param spark_df: Input Spark DataFrame
    :type spark_df: Any
    :return: Converted Polars DataFrame
    :rtype: pl.DataFrame
    """
    # noinspection PyProtectedMember,PyArgumentList
    polars_df = pl.from_arrow(pa.Table.from_batches(spark_df._collect_as_arrow()))

    return polars_df


def polars_gbq(query: str) -> pl.DataFrame:
    """
    Execute a SQL query on Google BigQuery and return result as Polars DataFrame.
    
    :param query: BigQuery SQL query string
    :type query: str
    :return: Query results as Polars DataFrame
    :rtype: pl.DataFrame
    """
    client = bigquery.Client()
    query_job = client.query(query)
    rows = query_job.result()
    df = pl.from_arrow(rows.to_arrow())

    return df


def get_phecode_mapping_table(
    phecode_version: str, 
    icd_version: str, 
    phecode_map_file_path: str | None, 
    keep_all_columns: bool = True
) -> pl.DataFrame:
    """
    Load phecode mapping table based on version specifications.
    
    :param phecode_version: Phecode version to use ("X" or "1.2")
    :type phecode_version: str
    :param icd_version: ICD version ("US", "WHO", or "custom")
    :type icd_version: str
    :param phecode_map_file_path: Path to custom phecode mapping file (required if icd_version="custom")
    :type phecode_map_file_path: str | None
    :param keep_all_columns: Whether to keep all columns in the mapping table
    :type keep_all_columns: bool
    :return: Phecode mapping table as Polars DataFrame
    :rtype: pl.DataFrame
    """
    # load a phecode mapping file by version or by custom path
    phetk_dir = os.path.dirname(__file__)
    final_file_path = os.path.join(phetk_dir, "phecode")
    path_suffix = ""
    if phecode_version == "X":
        if icd_version == "US":
            path_suffix = "phecodeX.csv"
        elif icd_version == "WHO":
            path_suffix = "phecodeX_WHO.csv"
        elif icd_version == "custom":
            if phecode_map_file_path is None:
                print("Please provide phecode_map_path for custom icd_version")
                sys.exit(1)
        else:
            print("Invalid icd_version. Available icd_version values are US, WHO and custom.")
            sys.exit(1)
        if phecode_map_file_path is None:
            final_file_path = os.path.join(final_file_path, path_suffix)
        else:
            final_file_path = phecode_map_file_path
        # noinspection PyTypeChecker
        phecode_df = pl.read_csv(final_file_path,
                                 dtypes={"phecode": str,
                                         "ICD": str,
                                         "flag": pl.Int8,
                                         "code_val": float})
        if not keep_all_columns:
            phecode_df = phecode_df[["phecode", "ICD", "flag"]]
    elif phecode_version == "1.2":
        if icd_version == "US":
            path_suffix = "phecode12.csv"
        elif icd_version == "WHO":
            print("PheTK does not support mapping ICD-10 (WHO version) to phecode 1.2")
            sys.exit(1)
        elif icd_version == "custom":
            if phecode_map_file_path is None:
                print("Please provide phecode_map_path for custom icd_version")
                sys.exit(1)
        else:
            print("Invalid icd_version. Available icd_version values are US, WHO and custom.")
            sys.exit(1)
        if phecode_map_file_path is None:
            final_file_path = os.path.join(final_file_path, path_suffix)
        else:
            final_file_path = phecode_map_file_path
        # noinspection PyTypeChecker
        phecode_df = pl.read_csv(final_file_path,
                                 dtypes={"phecode": str,
                                         "ICD": str,
                                         "flag": pl.Int8,
                                         "exclude_range": str,
                                         "phecode_unrolled": str})
        if not keep_all_columns:
            phecode_df = phecode_df[["phecode_unrolled", "ICD", "flag"]]
    else:
        print("Unsupported phecode version. Supports phecode \"1.2\" and \"X\".")
        sys.exit(1)

    return phecode_df


def generate_chunk_queries(query_function, ds: str, id_list: list, chunk_size: int = 1000) -> list[str]:
    """
    Generate a list of queries using a query generating function, each takes a chunk of IDs as input.
    
    :param query_function: Query function that generates SQL queries from chunks
    :type query_function: callable
    :param ds: Input dataset identifier
    :type ds: str
    :param id_list: List of IDs to chunk and process
    :type id_list: list
    :param chunk_size: Size of each chunk for processing
    :type chunk_size: int
    :return: List of generated SQL queries
    :rtype: list[str]
    """
    chunks = [
        list(id_list)[i * chunk_size:(i + 1) * chunk_size] for i in
        range((len(id_list) // chunk_size) + 1)
    ]

    print("Generating queries...")
    with ThreadPoolExecutor() as executor:
        jobs = [
            executor.submit(
                query_function,
                ds,
                tuple(chunk)
            ) for chunk in chunks
        ]
        result_list = [job.result() for job in tqdm(as_completed(jobs), total=len(chunks))]

    # process result
    query_list = [result for result in result_list if result is not None]

    print("Done!")
    print()

    return query_list


def polars_gbq_chunk(query_list: list[str]) -> pl.DataFrame:
    """
    Execute a list of BigQuery SQL queries and merge results into a single DataFrame.
    
    :param query_list: List of SQL query strings to execute
    :type query_list: list[str]
    :return: Final merged Polars DataFrame with unique rows
    :rtype: pl.DataFrame
    """

    print("Querying data...")
    with ThreadPoolExecutor() as executor:
        jobs = [
            executor.submit(
                polars_gbq,
                query
            ) for query in query_list
        ]
        result_list = [job.result() for job in tqdm(as_completed(jobs), total=len(query_list))]

    # process result
    result_list = [result for result in result_list if result is not None]
    final_result = result_list[0]  # assign the first dataframe in the result list as the final result and then concat with the rest
    for i in range(1, len(query_list)):
        final_result = pl.concat([final_result, result_list[i]])
    final_result = final_result.unique()
    print("Done!")
    print()

    return final_result

def detect_delimiter(file_path: str) -> str:
    """
    Detect delimiter (comma or tab) in a CSV/TSV file.
    Supports both local files and Google Cloud Storage paths.
    
    :param file_path: Path to the file (local path or gs:// URL)
    :type file_path: str
    :return: Detected delimiter (',' or '\t')
    :rtype: str
    """
    # Check if it's a GCP bucket path and if we're running in dsub environment
    if file_path.startswith('gs://'):
        # Check if we're in a dsub worker (the file might be locally mounted)
        # dsub typically mounts input gs:// files to /mnt/data/input/gs/
        local_path = file_path.replace('gs://', '/mnt/data/input/gs/')
        
        # Try local file first (dsub environment)
        if os.path.exists(local_path):
            file_path = local_path
        else:
            # Fallback to GCS API (non-dsub environment)
            try:
                # Import storage only when needed
                from google.cloud import storage
                
                # Parse bucket and blob name
                path_parts = file_path[5:].split('/', 1)  # Remove 'gs://' prefix
                bucket_name = path_parts[0]
                blob_name = path_parts[1]
                
                # Read first line from GCP bucket
                client = storage.Client()
                bucket = client.bucket(bucket_name)
                blob = bucket.blob(blob_name)
                
                # Read first 1024 bytes to get first line
                first_chunk = blob.download_as_bytes(start=0, end=1024).decode('utf-8')
                first_line = first_chunk.split('\n')[0]
                
                if "\t" in first_line:
                    return "\t"
                elif "," in first_line:
                    return ","
                else:
                    # fallback to sniffer with the chunk
                    sniffer = csv.Sniffer()
                    delimiter = sniffer.sniff(first_chunk).delimiter
                    
                    if delimiter == ",":
                        return ","
                    elif delimiter == "\t":
                        return "\t"
                    else:
                        print(f"Error: File must be CSV or TSV format. Detected delimiter: '{delimiter}'")
                        sys.exit(1)
            except Exception as e:
                print(f"Error accessing GCS file {file_path}: {e}")
                sys.exit(1)
    
    # Local file handling (either original local path or converted GCS path)
    try:
        with open(file_path, "r") as file:
            first_line = file.readline()
            if "\t" in first_line:
                return "\t"
            elif "," in first_line:
                return ","
            else:
                # fallback to sniffer
                file.seek(0)
                sample = file.read(1024)
                sniffer = csv.Sniffer()
                delimiter = sniffer.sniff(sample).delimiter
                
                if delimiter == ",":
                    return ","
                elif delimiter == "\t":
                    return "\t"
                else:
                    print(f"Error: File must be CSV or TSV format. Detected delimiter: '{delimiter}'")
                    sys.exit(1)
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        sys.exit(1)

def has_overlapping_values(d: dict) -> bool:
    """
    Check if any values in a dictionary have overlapping elements.
    
    :param d: Dictionary with values that can be single items or lists
    :type d: dict
    :return: True if any values overlap, False otherwise
    :rtype: bool
    """
    # Convert all values to sets, handling both single items and lists
    sets = []
    for value in d.values():
        if isinstance(value, list):
            sets.append(set(value))
        else:
            sets.append({value})

    # Check all pairs for intersection
    for set1, set2 in combinations(sets, 2):
        if set1 & set2:  # an intersection is not empty
            return True

    return False

def generate_sh_script(script_name: str, commands: list[str]) -> None:
    """
    Generate an executable bash script with given commands.
    
    :param script_name: Name of the script file to create
    :type script_name: str
    :param commands: List of commands to include in the script
    :type commands: list[str]
    :return: None
    :rtype: None
    """
    with open(script_name, 'w') as f:
        f.write("#!/bin/bash\n")  # Shebang line for bash
        for command in commands:
            f.write(command + "\n")

    # Make script executable
    os.chmod(script_name, 0o755)

    print(f"Generated script: {script_name}")

def monitor_cpu_usage_link() -> None:
    """
    Generate and print a Google Cloud Console link for monitoring CPU utilization.
    Uses the GOOGLE_PROJECT environment variable to construct the link.
    
    :return: None
    :rtype: None
    """
    cpu_utilization = (
        f'https://console.cloud.google.com/monitoring/metrics-explorer?authuser=0&project='
        f'{os.getenv("GOOGLE_PROJECT")}&pageState=%7B%22xyChart%22:%7B%22dataSets%22:%5B%7B%22timeSeriesFilter%22:'
        f'%7B%22filter%22:%22metric.type%3D%5C%22compute.googleapis.com%2Finstance%2Fcpu%2Futilization%5C%22%20resource.'
        f'type%3D%5C%22gce_instance%5C%22%22,%22minAlignmentPeriod%22:%2260s%22,%22aggregations%22:'
        f'%5B%7B%22perSeriesAligner%22:%22ALIGN_MEAN%22,%22crossSeriesReducer%22:'
        f'%22REDUCE_NONE%22,%22alignmentPeriod%22:%2260s%22,%22groupByFields%22:'
        f'%5B%5D%7D,%7B%22crossSeriesReducer%22:%22REDUCE_NONE%22,%22alignmentPeriod%22:'
        f'%2260s%22,%22groupByFields%22:%5B%5D%7D%5D%7D,%22targetAxis%22:%22Y1%22,%22plotType%22:'
        f'%22LINE%22%7D%5D,%22options%22:%7B%22mode%22:%22COLOR%22%7D,%22constantLines%22:'
        f'%5B%5D,%22timeshiftDuration%22:%220s%22,%22y1Axis%22:%7B%22label%22:%22y1Axis%22,%22scale%22:'
        f'%22LINEAR%22%7D%7D,%22isAutoRefresh%22:true,%22timeSelection%22:%7B%22timeRange%22:%221h%22%7D%7D'
    )
    print(f'To see the CPU utilization of your Cloud analysis environment, click on this link {cpu_utilization}')

def check_cpu_idle(
    cpu_threshold: float,
    low_cpu_start_time: float | None,
    time_threshold: int,
    verbose: bool = True
) -> tuple[bool, float | None]:
    """
    Check if CPU usage is below threshold and track idle time.
    
    :param cpu_threshold: CPU usage threshold percentage
    :type cpu_threshold: float
    :param low_cpu_start_time: When CPU first dropped below threshold (None if not tracking)
    :type low_cpu_start_time: float | None
    :param time_threshold: Maximum time CPU can stay below threshold (seconds)
    :type time_threshold: int
    :param verbose: Whether to print status messages
    :type verbose: bool
    :return: Tuple of (should_kill, new_low_cpu_start_time)
    :rtype: tuple[bool, float | None]
    """
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        current_time = time.time()
        
        if cpu_percent < cpu_threshold:
            if low_cpu_start_time is None:
                low_cpu_start_time = current_time
                if verbose:
                    print(f"CPU usage dropped below {cpu_threshold}% at {time.strftime('%H:%M:%S')}")
                return False, low_cpu_start_time
            else:
                elapsed_time = current_time - low_cpu_start_time
                if elapsed_time >= time_threshold:
                    if verbose:
                        print(f"CPU usage below {cpu_threshold}% for {time_threshold}s. Should kill job.")
                    return True, low_cpu_start_time
                return False, low_cpu_start_time
        else:
            if low_cpu_start_time is not None:
                if verbose:
                    print(f"CPU usage recovered to {cpu_percent:.1f}%. Resetting timer.")
                return False, None
            return False, low_cpu_start_time
    except Exception as e:
        if verbose:
            print(f"Error in CPU monitoring: {e}")
        return False, low_cpu_start_time

def print_banner(text: str, char: str = "~") -> None:
    """
    Print a centered banner with dynamic width based on terminal size.
    
    :param text: Text to display in the banner
    :type text: str
    :param char: Character to use for the banner (default: ~)
    :type char: str
    :return: None
    :rtype: None
    """
    try:
        terminal_width = os.get_terminal_size().columns
    except (OSError, AttributeError):
        # Fallback for notebooks or environments without terminal
        terminal_width = 80
    
    # Calculate padding around text (4 spaces on each side)
    padding = max(0, (terminal_width - len(text) - 8) // 2)
    remaining = max(0, terminal_width - padding - len(text) - 8)
    
    banner = char * padding + "    " + text + "    " + char * remaining
    print(banner)


def save_pickle_object(obj, file_path: str) -> None:
    """
    Save a Python object as a pickle file.
    
    :param obj: Python object to save
    :type obj: Any
    :param file_path: Path where to save the pickle file
    :type file_path: str
    :return: None
    :rtype: None
    """
    import pickle
    
    with open(file_path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle_object(file_path: str):
    """
    Load a Python object from a pickle file.
    
    :param file_path: Path to the pickle file to load
    :type file_path: str
    :return: Loaded Python object
    :rtype: Any
    """
    import pickle
    
    with open(file_path, "rb") as f:
        return pickle.load(f)


def load_dsub_instance(pickle_file_path: str):
    """
    Load a previously saved dsub instance from pickle file.
    
    :param pickle_file_path: Path to the pickle file containing saved dsub instance
    :type pickle_file_path: str
    :return: Loaded dsub instance with all methods available
    :rtype: Any
    """
    dsub_instance = load_pickle_object(pickle_file_path)
    print(f"Dsub instance loaded from '{pickle_file_path}'")
    print(f"Job ID: {dsub_instance.job_id}")
    print(f"Job Name: {dsub_instance.job_name}")
    print("Use dsub_instance.check_status(streaming=True) to monitor job progress")
    print("Available methods: check_status(), view_log(), kill()")
    return dsub_instance


def sample_tsv_file(file_path: str, sample_ratio: float = 0.1) -> None:
    """
    Generate a random sample of a TSV file using Polars.
    
    Preserves headers and TSV format. Saves the sampled file in the same
    directory with a suffix indicating the ratio and 'sample' tag.
    
    :param file_path: Path to the input TSV file to sample
    :type file_path: str
    :param sample_ratio: Ratio of rows to sample (0.1 = 10%, default: 0.1)
    :type sample_ratio: float
    :return: None
    :rtype: None
    """
    if not (0.0 < sample_ratio <= 1.0):
        raise ValueError("sample_ratio must be between 0.0 and 1.0")
    
    # Generate output file path
    file_dir = os.path.dirname(file_path)
    file_name = os.path.basename(file_path)
    name_parts = os.path.splitext(file_name)
    base_name = name_parts[0]
    extension = name_parts[1] if name_parts[1] else '.tsv'
    
    sample_file_path = os.path.join(
        file_dir, 
        f"{base_name}_sample_{sample_ratio*100:.1f}pct{extension}"
    )
    
    print(f"Using Polars for sampling...")
    
    # Detect delimiter
    delimiter = detect_delimiter(file_path)
    
    # Read the file with Polars
    df = pl.read_csv(
        file_path, 
        separator=delimiter, 
        try_parse_dates=True
    )
    total_rows = len(df)
    
    # Calculate sample size
    sample_size = max(1, int(total_rows * sample_ratio))
    
    print(f"Sampling {sample_size} rows from {total_rows} total rows ({sample_ratio*100:.1f}%)")
    
    # Sample the data
    sampled_df = df.sample(n=sample_size, seed=42)
    
    # Write the sampled data
    sampled_df.write_csv(sample_file_path, separator=delimiter)
    
    print(f"Sample file created: {sample_file_path}")
