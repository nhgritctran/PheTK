from concurrent.futures import ThreadPoolExecutor, as_completed
from google.cloud import bigquery
from itertools import combinations
from tqdm import tqdm

import argparse
import csv
import json
import os
import polars as pl
import psutil
import pyarrow as pa
import re
import subprocess
import sys
import time


# Cached result of Verily Workbench detection. Populated on first call to
# is_verily_workbench() and reused thereafter to avoid repeated `wb` CLI probes.
_VERILY_WORKBENCH_CACHED: bool | None = None


def is_verily_workbench() -> bool:
    """
    Detect whether the current process is running on Verily Workbench.

    Probes the `wb` CLI with `wb workspace describe`; a zero-exit response is
    treated as a positive detection. The result is cached at module level so
    subsequent calls are O(1). Tests can reset the cache by setting
    `phetk._utils._VERILY_WORKBENCH_CACHED = None`.

    This is a read-only check and does not mutate environment variables. It is
    intentionally separate from `setup_verily_env()` so that existing callers
    of the latter are unaffected.

    Returns:
        True if running on Verily Workbench, False otherwise (including when
        the `wb` CLI is not installed, times out, or returns non-zero).
    """
    global _VERILY_WORKBENCH_CACHED
    if _VERILY_WORKBENCH_CACHED is not None:
        return _VERILY_WORKBENCH_CACHED
    try:
        result = subprocess.run(
            ["wb", "workspace", "describe", "--format=json"],
            capture_output=True, text=True, timeout=30,
            stdin=subprocess.DEVNULL,
        )
        _VERILY_WORKBENCH_CACHED = (result.returncode == 0)
    except (FileNotFoundError, PermissionError, subprocess.TimeoutExpired):
        _VERILY_WORKBENCH_CACHED = False
    return _VERILY_WORKBENCH_CACHED


def setup_verily_env() -> None:
    """
    Set up environment variables for Verily Workbench VMs.

    On standard Verily VMs, uses the `wb` CLI to extract GOOGLE_PROJECT,
    GOOGLE_CLOUD_PROJECT, WORKSPACE_CDR, and WORKSPACE_BUCKET. On NEMO GPU
    VMs (where `wb` is Weights & Biases), falls back to `gcloud` for project
    and bucket discovery; WORKSPACE_CDR must be set manually in that case.

    If multiple CDR datasets are found, the latest version is selected based
    on the C{year}Q{quarter}R{release} naming convention.

    Silently returns on non-Verily platforms (e.g., AoU/Terra) where the `wb`
    CLI is not installed.
    """
    env_vars = [
        "GOOGLE_PROJECT",
        "GOOGLE_CLOUD_PROJECT",
        "WORKSPACE_CDR",
        "WORKSPACE_BUCKET",
        "WORKSPACE_REFERENCED_BUCKET",
    ]
    already_set = {v for v in env_vars if os.environ.get(v)}

    essential = {"GOOGLE_PROJECT", "WORKSPACE_CDR", "WORKSPACE_BUCKET"}
    if essential.issubset(already_set):
        print("Environment variables already set:")
        for v in env_vars:
            val = os.environ.get(v)
            if val:
                print(f"  {v}={val}")
        return

    try:
        # Detect Verily Workbench CLI
        result = subprocess.run(
            ["wb", "workspace", "describe", "--format=json"],
            capture_output=True, text=True, timeout=30,
            stdin=subprocess.DEVNULL,
        )
        if result.returncode != 0:
            # Check if `wb` is actually Weights & Biases (wandb) — e.g. on
            # NEMO GPU VMs. If so, fall back to gcloud for project + buckets.
            combined_output = (result.stdout + result.stderr).lower()
            if ("wandb" in combined_output
                    or "weights" in combined_output
                    or "no such command" in combined_output):
                _setup_verily_env_gcloud(env_vars, already_set)
                return

            # wb is likely the Verily CLI but the command failed.
            print(
                f"Warning: 'wb workspace describe' returned exit code "
                f"{result.returncode}."
            )
            if result.stderr:
                print(f"  stderr: {result.stderr.strip()}")
            print(
                "  The wb CLI may not be logged in or no workspace is set.\n"
                "  Try running:\n"
                "    wb auth login\n"
                "    wb workspace set --id=<your-workspace-id>\n"
                "  Then restart the kernel and try again.\n"
                "  Environment variables were NOT set automatically."
            )
            return

        # --- Standard Verily VM path (wb CLI available) ---
        print("Verily Workbench detected. Checking environment variables...")
        workspace_info = json.loads(result.stdout)

        # Get Google project ID
        if "GOOGLE_PROJECT" not in already_set or "GOOGLE_CLOUD_PROJECT" not in already_set:
            project_id = workspace_info.get("googleProjectId")
            if project_id:
                if "GOOGLE_PROJECT" not in already_set:
                    os.environ["GOOGLE_PROJECT"] = project_id
                if "GOOGLE_CLOUD_PROJECT" not in already_set:
                    os.environ["GOOGLE_CLOUD_PROJECT"] = project_id

        # Get WORKSPACE_CDR and WORKSPACE_BUCKET from wb resource list
        needs_cdr = "WORKSPACE_CDR" not in already_set
        needs_bucket = "WORKSPACE_BUCKET" not in already_set
        if needs_cdr or needs_bucket:
            result = subprocess.run(
                ["wb", "resource", "list", "--format=json"],
                capture_output=True, text=True, timeout=30,
                stdin=subprocess.DEVNULL,
            )
            if result.returncode != 0:
                print(f"Warning: 'wb resource list' failed (exit code {result.returncode}).")
                if result.stderr:
                    print(f"  {result.stderr.strip()}")
            else:
                resources = json.loads(result.stdout)

                if not resources:
                    print("Warning: `wb resource list` returned an empty list.")
                    print(
                        "  This workspace may not have any resources added yet.\n"
                        "  Add the CDR dataset and create a GCS bucket, or set\n"
                        "  environment variables manually:\n"
                        "    os.environ['WORKSPACE_CDR'] = '<project_id>.<dataset_id>'\n"
                        "    os.environ['WORKSPACE_BUCKET'] = 'gs://<bucket-name>'"
                    )

                # Parse resources
                cdr_candidates = []
                all_bq_datasets = []
                gcs_buckets = []
                for resource in resources:
                    resource_type = resource.get("resourceType", "")

                    if resource_type in ("BQ_DATASET", "BIGQUERY_DATASET"):
                        dataset_id = resource.get("datasetId", "")
                        project_id = resource.get("projectId", "")
                        all_bq_datasets.append((resource_type, project_id, dataset_id))
                        if needs_cdr:
                            match = re.match(r"^C(\d{4})Q(\d+)R(\d+)$", dataset_id)
                            if match:
                                sort_key = (int(match.group(1)), int(match.group(2)), int(match.group(3)))
                                cdr_candidates.append((sort_key, project_id, dataset_id))

                    elif resource_type == "GCS_BUCKET":
                        bucket_name = resource.get("bucketName") or ""
                        resource_id = resource.get("id") or resource.get("name") or ""
                        stewardship = resource.get("stewardshipType") or ""
                        if bucket_name:
                            gcs_buckets.append((resource_id, bucket_name, stewardship))

                # CDR
                if needs_cdr:
                    if cdr_candidates:
                        cdr_candidates.sort()
                        _, best_project, best_dataset = cdr_candidates[-1]
                        os.environ["WORKSPACE_CDR"] = f"{best_project}.{best_dataset}"
                    else:
                        print("Warning: No CDR dataset matching pattern C{year}Q{quarter}R{release} found.")
                        if all_bq_datasets:
                            print("  BigQuery datasets found (none matched CDR pattern):")
                            for rtype, pid, did in all_bq_datasets:
                                print(f"    type={rtype}  project={pid}  dataset={did}")
                        print(
                            "  To set WORKSPACE_CDR manually:\n"
                            "    os.environ['WORKSPACE_CDR'] = '<project_id>.<dataset_id>'"
                        )

                # Buckets — user-created CONTROLLED only, skip all dataproc-*
                if needs_bucket:
                    _classify_wb_buckets(gcs_buckets, resources)

        # Print summary
        _print_env_summary(env_vars, already_set)

    except FileNotFoundError:
        # wb CLI not installed — not on Verily Workbench (e.g. AoU/Terra).
        # Stay silent; this is the normal path for non-Verily platforms.
        return
    except PermissionError:
        print(
            "Warning: wb CLI found but not executable (PermissionError). "
            "Environment variables were NOT set automatically."
        )
        return
    except subprocess.TimeoutExpired:
        print(
            "Warning: 'wb workspace describe' timed out after 30 seconds. "
            "Environment variables were NOT set automatically."
        )
        return
    except json.JSONDecodeError as e:
        print(
            f"Warning: could not parse wb CLI output (JSONDecodeError: {e}). "
            f"Environment variables were NOT set automatically."
        )
        return
    except KeyError as e:
        print(
            f"Warning: unexpected format in wb output — missing key {e}. "
            f"Environment variables were NOT set automatically."
        )
        return


def _classify_wb_buckets(gcs_buckets: list, resources: list) -> None:
    """
    Classify GCS buckets from ``wb resource list`` and set env vars.

    Skips ``dataproc-*`` and ``vwb-aou-*`` buckets. Remaining buckets are
    split by stewardship type:

    - CONTROLLED → ``WORKSPACE_BUCKET`` (or ``WORKSPACE_BUCKET1``, ``2``, ...)
    - REFERENCED → ``WORKSPACE_REFERENCED_BUCKET`` (or numbered)

    Args:
        gcs_buckets: List of (resource_id, bucket_name, stewardship) tuples.
        resources: Full resource list (used for diagnostics if no buckets found).
    """
    skip_prefixes = ("dataproc-", "vwb-aou-")
    controlled = [
        (rid, bn) for rid, bn, stew in gcs_buckets
        if stew == "CONTROLLED"
        and not any(rid.lower().startswith(p) for p in skip_prefixes)
    ]
    referenced = [
        (rid, bn) for rid, bn, stew in gcs_buckets
        if stew == "REFERENCED"
        and not any(rid.lower().startswith(p) for p in skip_prefixes)
    ]

    # --- CONTROLLED → WORKSPACE_BUCKET ---
    if len(controlled) == 1:
        os.environ["WORKSPACE_BUCKET"] = f"gs://{controlled[0][1]}"
    elif len(controlled) > 1:
        for i, (rid, bn) in enumerate(controlled, 1):
            os.environ[f"WORKSPACE_BUCKET{i}"] = f"gs://{bn}"
        print(
            "  Multiple controlled buckets found. Set WORKSPACE_BUCKET to "
            "the one you want to use, e.g.:\n"
            "    os.environ['WORKSPACE_BUCKET'] = os.environ['WORKSPACE_BUCKET1']"
        )

    # --- REFERENCED → WORKSPACE_REFERENCED_BUCKET ---
    if len(referenced) == 1:
        os.environ["WORKSPACE_REFERENCED_BUCKET"] = f"gs://{referenced[0][1]}"
    elif len(referenced) > 1:
        for i, (rid, bn) in enumerate(referenced, 1):
            os.environ[f"WORKSPACE_REFERENCED_BUCKET{i}"] = f"gs://{bn}"
        print(
            "  Multiple referenced buckets found. Set "
            "WORKSPACE_REFERENCED_BUCKET to the one you want to use, e.g.:\n"
            "    os.environ['WORKSPACE_REFERENCED_BUCKET'] = "
            "os.environ['WORKSPACE_REFERENCED_BUCKET1']"
        )

    # --- No user buckets at all ---
    if not controlled and not referenced:
        if not gcs_buckets:
            resource_types_seen = {
                r.get("resourceType", "<none>") for r in resources
            }
            print(
                "Warning: No GCS_BUCKET resources found in `wb resource list`.\n"
                f"  Resource types found: {', '.join(sorted(resource_types_seen)) or 'none'}\n"
                "  Create a GCS bucket in this workspace, or set manually:\n"
                "    os.environ['WORKSPACE_BUCKET'] = 'gs://<bucket-name>'"
            )
        else:
            print(
                "Warning: No user-created GCS bucket found.\n"
                "  Create a GCS bucket in this workspace, or set manually:\n"
                "    os.environ['WORKSPACE_BUCKET'] = 'gs://<bucket-name>'"
            )


def _classify_gcloud_buckets(bucket_names: list) -> None:
    """
    Classify GCS buckets from ``gcloud storage ls`` and set WORKSPACE_BUCKET.

    ``gcloud storage ls`` only returns CONTROLLED buckets (not referenced).
    Skips all ``dataproc-*``, ``cloned-*``, and ``vwb-aou-*`` buckets.
    Remaining ones are user-created and become WORKSPACE_BUCKET. If multiple
    exist, numbered variables are set instead.

    Args:
        bucket_names: List of bucket name strings (without ``gs://`` prefix).
    """
    skip_prefixes = ("dataproc-", "cloned-", "vwb-aou-")
    user_candidates = [
        bn for bn in bucket_names
        if not any(bn.lower().startswith(p) for p in skip_prefixes)
    ]

    if len(user_candidates) == 1:
        os.environ["WORKSPACE_BUCKET"] = f"gs://{user_candidates[0]}"
    elif len(user_candidates) > 1:
        print("Multiple user-created GCS buckets found:")
        for i, bn in enumerate(user_candidates, 1):
            os.environ[f"WORKSPACE_BUCKET{i}"] = f"gs://{bn}"
            print(f"  WORKSPACE_BUCKET{i}=gs://{bn}")
        print(
            "  Set WORKSPACE_BUCKET to the one you want to use, e.g.:\n"
            "    os.environ['WORKSPACE_BUCKET'] = os.environ['WORKSPACE_BUCKET1']"
        )
    else:
        print(
            "Warning: No user-created GCS bucket found.\n"
            "  Create a GCS bucket, or set manually:\n"
            "    os.environ['WORKSPACE_BUCKET'] = 'gs://<bucket-name>'"
        )


def _setup_verily_env_gcloud(env_vars: list, already_set: set) -> None:
    """
    Fallback for VMs where the Verily Workbench CLI is not available (e.g.
    NEMO GPU VMs where ``wb`` is Weights & Biases). Uses ``gcloud`` for
    project and bucket discovery. WORKSPACE_CDR cannot be auto-detected
    without the ``wb`` CLI and must be set manually.

    Args:
        env_vars: List of environment variable names to track.
        already_set: Set of variable names already present in the environment.
    """
    print("Attempting to set environment variables using gcloud...")

    # GOOGLE_PROJECT via gcloud
    if "GOOGLE_PROJECT" not in already_set or "GOOGLE_CLOUD_PROJECT" not in already_set:
        try:
            result = subprocess.run(
                ["gcloud", "config", "get-value", "project"],
                capture_output=True, text=True, timeout=15,
                stdin=subprocess.DEVNULL,
            )
            project_id = result.stdout.strip()
            if result.returncode == 0 and project_id:
                if "GOOGLE_PROJECT" not in already_set:
                    os.environ["GOOGLE_PROJECT"] = project_id
                if "GOOGLE_CLOUD_PROJECT" not in already_set:
                    os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
            else:
                print(
                    "Warning: could not determine project from gcloud.\n"
                    "  Set manually: os.environ['GOOGLE_PROJECT'] = '<project-id>'"
                )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            print(
                "Warning: gcloud CLI not available.\n"
                "  Set manually: os.environ['GOOGLE_PROJECT'] = '<project-id>'"
            )

    # WORKSPACE_CDR — cannot be auto-detected without wb resource list
    if "WORKSPACE_CDR" not in already_set:
        print(
            "WORKSPACE_CDR must be set manually:\n"
            "  os.environ['WORKSPACE_CDR'] = '<project_id>.<dataset_id>'"
        )

    # WORKSPACE_BUCKET via gcloud storage ls
    if "WORKSPACE_BUCKET" not in already_set:
        project_id = os.environ.get("GOOGLE_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT")
        if project_id:
            try:
                result = subprocess.run(
                    ["gcloud", "storage", "ls", f"--project={project_id}"],
                    capture_output=True, text=True, timeout=30,
                    stdin=subprocess.DEVNULL,
                )
                if result.returncode == 0 and result.stdout.strip():
                    # Parse bucket names: "gs://bucket-name/" → "bucket-name"
                    bucket_names = [
                        line.strip().removeprefix("gs://").rstrip("/")
                        for line in result.stdout.strip().splitlines()
                        if line.strip().startswith("gs://")
                    ]
                    _classify_gcloud_buckets(bucket_names)
                else:
                    print(
                        "Warning: `gcloud storage ls` returned no buckets.\n"
                        "  Create a GCS bucket, or set manually:\n"
                        "    os.environ['WORKSPACE_BUCKET'] = 'gs://<bucket-name>'"
                    )
            except (FileNotFoundError, subprocess.TimeoutExpired):
                print(
                    "Warning: gcloud CLI not available for bucket listing.\n"
                    "  Set manually: os.environ['WORKSPACE_BUCKET'] = 'gs://<bucket-name>'"
                )
        else:
            print(
                "Warning: GOOGLE_PROJECT not set; cannot list buckets.\n"
                "  Set manually: os.environ['WORKSPACE_BUCKET'] = 'gs://<bucket-name>'"
            )

    _print_env_summary(env_vars, already_set)


def _print_env_summary(env_vars: list, already_set: set) -> None:
    """Print summary of which environment variables were set vs skipped."""
    newly_set = []
    skipped = []
    for v in env_vars:
        if v in already_set:
            skipped.append(f"  {v}={os.environ[v]}")
        elif os.environ.get(v):
            newly_set.append(f"  {v}={os.environ[v]}")
        # Check for numbered variants (e.g. WORKSPACE_BUCKET1, WORKSPACE_BUCKET2)
        for i in range(1, 20):
            numbered = f"{v}{i}"
            val = os.environ.get(numbered)
            if val:
                newly_set.append(f"  {numbered}={val}")
            else:
                break
    if newly_set:
        print("Environment variables set:")
        for line in newly_set:
            print(line)
    if skipped:
        print("Already set (skipped):")
        for line in skipped:
            print(line)


def str_to_bool(v) -> bool:
    """
    Convert string to boolean for argparse.

    Args:
        v: Input value to convert to boolean.

    Returns:
        Boolean value.
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

    Args:
        df: Input dataframe object (pandas DataFrame or polars DataFrame).

    Returns:
        Polars DataFrame.
    """
    if not isinstance(df, pl.DataFrame):
        return pl.from_pandas(df)
    else:
        return df


def spark_to_polars(spark_df) -> pl.DataFrame:
    """
    Convert Spark DataFrame to Polars DataFrame.

    Args:
        spark_df: Input Spark DataFrame.

    Returns:
        Converted Polars DataFrame.
    """
    # noinspection PyProtectedMember,PyArgumentList
    polars_df = pl.from_arrow(pa.Table.from_batches(spark_df._collect_as_arrow()))

    return polars_df


def polars_gbq(query: str) -> pl.DataFrame:
    """
    Execute a SQL query on Google BigQuery and return result as Polars DataFrame.

    Args:
        query: BigQuery SQL query string.

    Returns:
        Query results as Polars DataFrame.
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

    Args:
        phecode_version: Phecode version to use ("X" or "1.2").
        icd_version: ICD version ("US", "WHO", or "custom").
        phecode_map_file_path: Path to custom phecode mapping file (required if icd_version="custom").
        keep_all_columns: Whether to keep all columns in the mapping table.

    Returns:
        Phecode mapping table as Polars DataFrame.
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

    Args:
        query_function: Query function that generates SQL queries from chunks.
        ds: Input dataset identifier.
        id_list: List of IDs to chunk and process.
        chunk_size: Size of each chunk for processing.

    Returns:
        List of generated SQL queries.
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

    Args:
        query_list: List of SQL query strings to execute.

    Returns:
        Final merged Polars DataFrame with unique rows.
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

    Args:
        file_path: Path to the file (local path or gs:// URL).

    Returns:
        Detected delimiter (',' or '\\t').
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

    Args:
        d: Dictionary with values that can be single items or lists.

    Returns:
        True if any values overlap, False otherwise.
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

    Args:
        script_name: Name of the script file to create.
        commands: List of commands to include in the script.
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

    Args:
        cpu_threshold: CPU usage threshold percentage.
        low_cpu_start_time: When CPU first dropped below threshold (None if not tracking).
        time_threshold: Maximum time CPU can stay below threshold (seconds).
        verbose: Whether to print status messages.

    Returns:
        Tuple of (should_kill, new_low_cpu_start_time).
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

    Args:
        text: Text to display in the banner.
        char: Character to use for the banner (default: ~).
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

    Args:
        obj: Python object to save.
        file_path: Path where to save the pickle file.
    """
    import pickle
    
    with open(file_path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle_object(file_path: str):
    """
    Load a Python object from a pickle file.

    Args:
        file_path: Path to the pickle file to load.

    Returns:
        Loaded Python object.
    """
    import pickle
    
    with open(file_path, "rb") as f:
        return pickle.load(f)


def load_dsub_instance(pickle_file_path: str):
    """
    Load a previously saved dsub instance from pickle file.

    Args:
        pickle_file_path: Path to the pickle file containing saved dsub instance.

    Returns:
        Loaded dsub instance with all methods available.
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

    Args:
        file_path: Path to the input TSV file to sample.
        sample_ratio: Ratio of rows to sample (0.1 = 10%, default: 0.1).
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
