from concurrent.futures import ThreadPoolExecutor, as_completed
from google.cloud import bigquery
from itertools import combinations
from tqdm import tqdm

import csv
import os
import polars as pl
import pyarrow as pa
import sys


def to_polars(df):
    """
    Check and convert pandas dataframe object to polars dataframe, if applicable
    :param df: dataframe object
    :return: polars dataframe
    """
    if not isinstance(df, pl.DataFrame):
        return pl.from_pandas(df)
    else:
        return df


def spark_to_polars(spark_df):
    """
    Convert spark df to polars df
    :param spark_df: spark df
    :return: polars df
    """
    # noinspection PyProtectedMember,PyArgumentList
    polars_df = pl.from_arrow(pa.Table.from_batches(spark_df._collect_as_arrow()))

    return polars_df


def polars_gbq(query):
    """
    Take a SQL query and return result as polars dataframe
    :param query: BigQuery SQL query
    :return: polars dataframe
    """
    client = bigquery.Client()
    query_job = client.query(query)
    rows = query_job.result()
    df = pl.from_arrow(rows.to_arrow())

    return df


def get_phecode_mapping_table(phecode_version, icd_version, phecode_map_file_path, keep_all_columns=True):
    """
    Load phecode mapping table
    :param phecode_version: defaults to "X"; the other option is "1.2"
    :param icd_version: defaults to "US"; the other options are "WHO" and "custom";
                        if "custom", users need to provide phecode_map_path
    :param phecode_map_file_path: path to custom phecode map table
    :param keep_all_columns: defaults to True
    :return: phecode mapping table as polars dataframe
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


def generate_chunk_queries(query_function, ds, id_list, chunk_size=1000):
    """
    Generate a list of queries using a query generating function, each takes a chunk of IDs as input
    :param query_function: query function
    :param ds: input dataset
    :param id_list: list of IDs
    :param chunk_size: chunk size
    :return: list of queries
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


def polars_gbq_chunk(query_list):
    """
    This takes a list of queries as input and generates a final merged dataframe from them.
    :param query_list: List of queries
    :return: Final merged polars dataframe
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

def detect_delimiter(file_path):
    """
    Detect delimiter (comma or tab) in a CSV/TSV file
    Supports both local files and Google Cloud Storage paths
    :param file_path: Path to the file (local or gs:// URL)
    :return: Detected delimiter (',' or '\t')
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

def has_overlapping_values(d):
    """
    Check if any values in a dictionary have overlapping elements
    :param d: Dictionary with values that can be single items or lists
    :return: True if any values overlap, False otherwise
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

def generate_sh_script(script_name, commands):
    """
    Generate an executable bash script with given commands
    :param script_name: Name of the script file to create
    :param commands: List of commands to include in the script
    """
    with open(script_name, 'w') as f:
        f.write("#!/bin/bash\n")  # Shebang line for bash
        for command in commands:
            f.write(command + "\n")

    # Make script executable
    os.chmod(script_name, 0o755)

    print(f"Generated script: {script_name}")

def monitor_cpu_usage_link():
    """
    Generate and print a Google Cloud Console link for monitoring CPU utilization
    Uses the GOOGLE_PROJECT environment variable to construct the link
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


def print_banner(text: str, char: str = "~") -> None:
    """
    Print a centered banner with dynamic width based on terminal size
    :param text: Text to display in the banner
    :param char: Character to use for the banner (default: ~)
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
