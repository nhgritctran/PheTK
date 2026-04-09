import os
import duckdb
import polars as pl
import psutil
import sys
# noinspection PyUnresolvedReferences,PyProtectedMember
from phetk import _queries, _utils


def _enable_line_buffered_stdout() -> None:
    """
    Force stdout to line-buffered so CLI progress prints show up immediately.

    Under non-TTY stdouts (dsub logs, piped terminals, captured subprocesses)
    Python defaults to block buffering, which makes phetk CLI runs look
    completely silent until the process exits. Switching to line buffering
    flushes on every newline -- cheap, safe, and makes `print(...)` Just Work.
    """
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        # older Python / exotic streams: fall back silently; callers that
        # really need a flush can still use print(..., flush=True).
        pass


def _auto_memory_limit_gb(fraction: float = 0.9, minimum_gb: int = 4) -> int:
    """
    Pick a DuckDB memory_limit based on currently-available RAM.

    Uses psutil.virtual_memory().available (not total) so already-resident
    objects like self.icd_events are accounted for. The default fraction
    (90% of available) leaves a thin margin for the OS and for the polars
    DataFrame we hand back after the join + aggregation -- the aggregated
    result is small, so this margin doesn't need to be large.
    """
    available_gb = psutil.virtual_memory().available // (1024 ** 3)
    return max(minimum_gb, int(available_gb * fraction))


class Phecode:
    """
    Extract ICD codes and map them to phecodes for phenome-wide association studies.
    
    Supports phecode versions 1.2 and X with ICD code extraction from All of Us OMOP
    database or custom ICD code tables. Handles ICD9CM and ICD10CM vocabularies and
    provides functionality for phecode counting, age calculation, and time-to-event analysis.
    """

    def __init__(
            self,
            platform: str = "aou",
            icd_file_path: str | None = None,
            gbq_dataset_id: str | None = None
    ):
        """
        Initialize Phecode object for ICD code extraction and phecode mapping.

        Sets up data source configuration and loads ICD code data either from
        All of Us OMOP database or custom file. Validates vocabulary types and
        creates ICD version flags for downstream processing.

        Args:
            platform: Data platform, supports "aou" (All of Us) or "custom".
            icd_file_path: Path to ICD table CSV/TSV file with columns "person_id", "ICD", "vocabulary_id".
            gbq_dataset_id: BigQuery dataset ID. Overrides WORKSPACE_CDR on AoU.
        """
        self.platform = platform

        if platform == "aou":
            _utils.setup_verily_env()
            if gbq_dataset_id is not None:
                self.cdr = gbq_dataset_id
            else:
                self.cdr = os.getenv("WORKSPACE_CDR")
            if self.cdr is None:
                print("WORKSPACE_CDR environment variable is not set. "
                      "On Verily Workbench, run phetk.setup_verily_env() first. "
                      "Or provide gbq_dataset_id in the constructor: "
                      "Phecode(gbq_dataset_id=\"your_dataset_id\"), "
                      "or use Phecode(platform=\"custom\", icd_file_path=\"...\") "
                      "for non-AoU environments.")
                sys.exit(1)
            self.icd_query = _queries.phecode_icd_query(self.cdr)
            print("Start querying ICD codes...", flush=True)
            self.icd_events = _utils.polars_gbq(self.icd_query)

        elif platform == "custom":
            self.cdr = gbq_dataset_id
            if icd_file_path is not None:
                print("Loading user's ICD data from file...", flush=True)
                sep = _utils.detect_delimiter(icd_file_path)
                self.icd_events = pl.read_csv(
                    icd_file_path,
                    separator=sep,
                    schema_overrides={"ICD": str}
                )
            else:
                print("icd_file_path is required for custom platform.")
                sys.exit(0)
        else:
            print("Invalid platform. Parameter platform only accepts \"aou\" (All of Us) or \"custom\".")
            sys.exit(0)

        # add flag column if not exist
        if "flag" not in self.icd_events.columns:
            self.icd_events = self.icd_events.with_columns(
                pl.when((pl.col("vocabulary_id") == "ICD9") |
                        (pl.col("vocabulary_id") == "ICD9CM"))
                .then(9)
                .when((pl.col("vocabulary_id") == "ICD10") |
                      (pl.col("vocabulary_id") == "ICD10CM"))
                .then(10)
                .otherwise(0)
                .alias("flag")
                .cast(pl.Int8)
            )
        else:
            self.icd_events = self.icd_events.with_columns(pl.col("flag").cast(pl.Int8))

        print("Done!", flush=True)

    def count_phecode(
            self,
            phecode_version: str = "X",
            icd_version: str = "US",
            phecode_map_file_path: str | None = None,
            output_file_path: str | None = None,
            engine: str = "duckdb",
            memory_limit: str | None = None,
    ) -> None:
        """
        Generate phecode counts from ICD code data.

        Maps ICD codes to phecodes using specified version and mapping table,
        aggregates counts per person-phecode combination, and calculates first
        event dates. Creates output file with person-level phecode statistics.

        Args:
            phecode_version: Phecode version to use, "X" or "1.2".
            icd_version: ICD mapping version, "US", "WHO", or "custom".
            phecode_map_file_path: Path to custom phecode mapping table.
            output_file_path: Path for output TSV file.
            engine: Execution engine for the ICD->phecode mapping step.
                "duckdb" (default): DuckDB performs the join + group_by, spilling
                to disk if it exceeds `memory_limit`. Handles large cohorts
                (e.g. full AoU v8) without OOM.
                "polars": in-memory polars join. Lowest overhead for
                small/medium cohorts but can OOM on very large cohorts.
            memory_limit: DuckDB memory limit (e.g. "64GB"). Only used when
                engine="duckdb". If None (default), auto-sized to ~90% of
                currently-available RAM, which keeps the join in memory on
                large AoU VMs (e.g. ~90 GB on a 104 GB machine) instead of
                forcing it to spill and crawl.

        Returns:
            Creates phecode counts TSV file with person_id, phecode, count, and first_event_date.
        """
        # load the phecode mapping file by version or by custom path
        phecode_df = _utils.get_phecode_mapping_table(
            phecode_version=phecode_version,
            icd_version=icd_version,
            phecode_map_file_path=phecode_map_file_path,
            keep_all_columns=False
        )

        phecode_version_string = phecode_version
        if phecode_version == "1.2":
            phecode_version_string = " " + phecode_version

        # resolve output path once (shared across engines)
        if output_file_path is None:
            file_path = "{0}_{1}_phecode{2}_counts.tsv".format(
                self.platform, icd_version, phecode_version.upper().replace(".", ""))
        else:
            file_path = output_file_path

        # shared: select only the needed columns (no clone -- polars returns a new frame,
        # self.icd_events is not mutated, so this is safe to reuse across engines and
        # across multiple count_phecode calls on the same instance)
        icd_events = self.icd_events.select(["person_id", "date", "ICD", "flag"])

        print()
        print(f"Mapping ICD codes to phecode{phecode_version_string}...", flush=True)

        if engine == "polars":
            # ---------- polars path (existing logic minus redundant clone) ----------
            if phecode_version == "X":
                phecode_counts = icd_events.join(phecode_df,
                                                 how="inner",
                                                 on=["ICD", "flag"])
            elif phecode_version == "1.2":
                phecode_counts = icd_events.join(phecode_df,
                                                 how="inner",
                                                 on=["ICD", "flag"])
                phecode_counts = phecode_counts.rename({"phecode_unrolled": "phecode"})
            else:
                phecode_counts = pl.DataFrame()

            if not phecode_counts.is_empty():
                phecode_counts = phecode_counts.group_by(
                    ["person_id", "phecode"]
                ).agg(
                    pl.len(), pl.col("date").min()
                ).rename(
                    {"len": "count", "date": "first_event_date"}
                )

        elif engine == "duckdb":
            # ---------- duckdb path ----------
            # DuckDB performs the row-multiplying join + aggregation. On a large
            # VM with a properly sized memory_limit this runs entirely in RAM;
            # if the intermediate exceeds memory_limit DuckDB spills to disk
            # instead of OOM-ing.
            #
            # We then hand the small aggregated result back to polars zero-copy
            # via Arrow and let polars's object-store backend handle the write
            # -- this is what makes gs:// / s3:// destinations work without
            # configuring DuckDB's httpfs extension and credentials.
            #
            # The aggregated result is bounded by the number of unique
            # (person_id, phecode) pairs, which is orders of magnitude smaller
            # than the join expansion, so it fits in RAM even when the join
            # itself does not.
            if memory_limit is None:
                resolved_memory_limit = f"{_auto_memory_limit_gb()}GB"
            else:
                resolved_memory_limit = memory_limit
            n_threads = psutil.cpu_count(logical=True) or 1
            print(
                f"  DuckDB engine: memory_limit={resolved_memory_limit}, "
                f"threads={n_threads}",
                flush=True,
            )
            with duckdb.connect() as con:
                con.execute(f"PRAGMA memory_limit='{resolved_memory_limit}'")
                con.execute(f"PRAGMA threads={n_threads}")
                con.register("events_view", icd_events)
                con.register("mapping_view", phecode_df)

                if phecode_version == "X":
                    # phecode mapping already has a "phecode" column for version X
                    sql = """
                        SELECT
                            e.person_id,
                            m.phecode          AS phecode,
                            COUNT(*)           AS count,
                            MIN(e.date)        AS first_event_date
                        FROM events_view AS e
                        INNER JOIN mapping_view AS m USING (ICD, flag)
                        GROUP BY e.person_id, m.phecode
                    """
                elif phecode_version == "1.2":
                    # phecode 1.2 mapping uses "phecode_unrolled" -- rename via SQL alias.
                    # GROUP BY references the raw column because SELECT aliases
                    # aren't visible in GROUP BY in standard SQL.
                    sql = """
                        SELECT
                            e.person_id,
                            m.phecode_unrolled AS phecode,
                            COUNT(*)           AS count,
                            MIN(e.date)        AS first_event_date
                        FROM events_view AS e
                        INNER JOIN mapping_view AS m USING (ICD, flag)
                        GROUP BY e.person_id, m.phecode_unrolled
                    """
                else:
                    sql = None

                if sql is not None:
                    # .pl() returns a polars DataFrame backed by an Arrow result;
                    # the buffers are owned by the polars frame and remain valid
                    # after the connection is closed.
                    phecode_counts = con.execute(sql).pl()
                else:
                    phecode_counts = pl.DataFrame()

        else:
            raise ValueError(f"engine must be 'polars' or 'duckdb', got {engine!r}")

        # shared write + reporting
        if not phecode_counts.is_empty():
            print()
            print(f"Mapping done. Writing output to {file_path}...", flush=True)
            # polars handles local paths and cloud paths (gs://, s3://, ...)
            # via its built-in object-store backend.
            phecode_counts.write_csv(file_path, separator="\t")
            print(f"Successfully generated phecode{phecode_version} counts for cohort participants!")
            print()
            print(f"Saved to\033[1m {file_path}\033[0m")
            print()
        else:
            print("\033[1mNo phecode count generated. Check your input data.\033[0m")
            print()

    def add_age_at_first_event(
            self,
            phecode_count_file_path: str,
            output_file_path: str | None = None,
    ) -> None:
        """
        Calculate age at first phecode event for each participant.

        Retrieves participant birth dates from OMOP database and calculates
        age at first phecode occurrence. Uses chunked processing for efficient
        handling of large datasets.

        Args:
            phecode_count_file_path: Path to phecode counts TSV file with person_id, phecode, and first_event_date columns.
            output_file_path: Path for output file with age calculations.

        Returns:
            Creates enhanced phecode counts TSV file with age_at_first_event column.
        """
        print("Calculating age at first event...")
        sep = _utils.detect_delimiter(phecode_count_file_path)
        phecode_counts = pl.read_csv(
            phecode_count_file_path,
            separator=sep,
            schema_overrides={"phecode": str, "first_event_date": pl.Date()}
        )

        participant_ids = phecode_counts["person_id"].unique().to_list()

        query_list = _utils.generate_chunk_queries(
            query_function=_queries.current_age_query,
            ds=self.cdr,
            id_list=participant_ids,
            chunk_size=1000
        )

        date_of_birth_df = _utils.polars_gbq_chunk(
            query_list=query_list,
        )
        # # old version without chunking
        # date_of_birth_df = _utils.polars_gbq(_queries.current_age_query(ds=self.cdr, participant_ids=participant_ids))

        print("Processing data...")

        date_of_birth_df = date_of_birth_df[["person_id", "date_of_birth"]]

        phecode_counts = phecode_counts.join(date_of_birth_df, how="inner", on=["person_id"])
        col_name = "age_at_first_event"
        phecode_counts = phecode_counts.with_columns(
            (
                (pl.col("first_event_date") - pl.col("date_of_birth")).dt.total_days()/365.2425
            ).alias(col_name)
        )
        phecode_counts = phecode_counts[["person_id", "phecode", "count", "first_event_date", "age_at_first_event"]]

        if output_file_path is None:
            output_file_path = phecode_count_file_path.replace(".tsv", "_with_age_at_first_event.tsv")
        phecode_counts.write_csv(output_file_path, separator="\t")
        print("Done!")
        print()
        print(f"Saved to\033[1m {output_file_path}\033[0m. "
              f"Age at first event column name is {col_name}.")
        print()

    @staticmethod
    def add_phecode_time_to_event(
            phecode_count_file_path: str,
            cohort_file_path: str,
            study_start_date_col: str,
            time_unit: str = "days",
            output_file_path: str | None = None,
    ) -> None:
        """
        Calculate time from study start to first phecode event for survival analysis.

        Computes time-to-event for each phecode relative to participant study
        start dates. Useful for Cox regression and time-to-event analyses in
        longitudinal cohort studies.

        Args:
            phecode_count_file_path: Path to phecode counts CSV/TSV file.
            cohort_file_path: Path to cohort CSV/TSV file with study start dates.
            study_start_date_col: Column name containing study start dates.
            time_unit: Time unit for calculations, "days" or "years".
            output_file_path: Path for output file with time calculations.

        Returns:
            Creates enhanced phecode counts TSV file with phecode_time_to_event column.
        """

        print("Calculating time to event for each phecode...")
        phecode_sep = _utils.detect_delimiter(phecode_count_file_path)
        phecode_counts = pl.read_csv(
            phecode_count_file_path,
            separator=phecode_sep,
            schema_overrides={"phecode": str, "first_event_date": pl.Date()}
        )
        cohort_sep = _utils.detect_delimiter(cohort_file_path)
        cohort_df = pl.read_csv(
            cohort_file_path,
            separator=cohort_sep,
            schema_overrides={study_start_date_col: pl.Date()}
        )
        cohort_df = cohort_df[["person_id", study_start_date_col]]

        phecode_counts = phecode_counts.join(cohort_df, how="inner", on=["person_id"])
        col_name = "phecode_time_to_event"
        if time_unit == "days":
            denominator = 1
        elif time_unit == "years":
            denominator = 365.2425
        else:
            raise ValueError("time_unit must be either 'days' or 'years'")

        phecode_counts = phecode_counts.with_columns(
            (
                    (pl.col("first_event_date") - pl.col(study_start_date_col)).dt.total_days() / denominator
            ).alias(col_name)
        )

        phecode_counts = phecode_counts[["person_id", "phecode", "count", "first_event_date", "phecode_time_to_event"]]

        if output_file_path is None:
            output_file_path = phecode_count_file_path.replace(".tsv", "_with_phecode_time_to_event.tsv")
        phecode_counts.write_csv(output_file_path, separator="\t")
        print("Done!")
        print()
        print(f"Saved to\033[1m {output_file_path}\033[0m. "
              f"Phecode time to event column name is {col_name}.")
        print()


def main_count_phecode():
    """Main entry point for count-phecode CLI command."""
    import argparse

    _enable_line_buffered_stdout()

    parser = argparse.ArgumentParser(
        description="Generate phecode counts from ICD code data"
    )
    
    # Platform arguments
    parser.add_argument("--platform", type=str, default="aou",
                        help="Data platform: 'aou' or 'custom' (default: aou)")
    parser.add_argument("--icd_file_path", type=str, default=None,
                        help="Path to ICD table CSV/TSV file (required for custom platform)")
    parser.add_argument("--gbq_dataset_id", type=str, default=None,
                        help="BigQuery dataset ID. Overrides WORKSPACE_CDR on AoU.")

    # Phecode mapping arguments
    parser.add_argument("--phecode_version", type=str, default="X",
                        help="Phecode version to use: 'X' or '1.2' (default: X)")
    parser.add_argument("--icd_version", type=str, default="US",
                        help="ICD mapping version: 'US', 'WHO', or 'custom' (default: US)")
    parser.add_argument("--phecode_map_file_path", type=str, default=None,
                        help="Path to custom phecode mapping table")
    
    # Output argument
    parser.add_argument("--output_file_path", "-o", type=str, default=None,
                        help="Path for output TSV file")

    # Engine argument
    parser.add_argument("--engine", type=str, default="duckdb", choices=["duckdb", "polars"],
                        help="Execution engine for ICD->phecode mapping: "
                             "'duckdb' (default, out-of-core, recommended for large cohorts) "
                             "or 'polars' (in-memory)")
    parser.add_argument("--memory_limit", type=str, default=None,
                        help="DuckDB memory limit (e.g. '64GB'). Only used with --engine=duckdb. "
                             "If omitted, auto-sized to ~90%% of available RAM.")

    args = parser.parse_args()

    # Create phecode instance
    phecode = Phecode(
        platform=args.platform,
        icd_file_path=args.icd_file_path,
        gbq_dataset_id=args.gbq_dataset_id
    )

    # Run count_phecode
    phecode.count_phecode(
        phecode_version=args.phecode_version,
        icd_version=args.icd_version,
        phecode_map_file_path=args.phecode_map_file_path,
        output_file_path=args.output_file_path,
        engine=args.engine,
        memory_limit=args.memory_limit,
    )


def main_add_age_at_first_event():
    """Main entry point for add-age-at-first-event CLI command."""
    import argparse

    _enable_line_buffered_stdout()

    parser = argparse.ArgumentParser(
        description="Calculate age at first phecode event for each participant"
    )
    
    # Required arguments
    parser.add_argument("--phecode_count_file_path", "-c", type=str, required=True,
                        help="Path to phecode counts TSV file with person_id, phecode, and first_event_date columns")
    
    # Platform arguments (for database access)
    parser.add_argument("--platform", type=str, default="aou",
                        help="Data platform: 'aou' or 'custom' (default: aou)")
    parser.add_argument("--icd_file_path", type=str, default=None,
                        help="Path to ICD table CSV/TSV file (required for custom platform)")
    parser.add_argument("--gbq_dataset_id", type=str, default=None,
                        help="BigQuery dataset ID. Overrides WORKSPACE_CDR on AoU.")

    # Output argument
    parser.add_argument("--output_file_path", "-o", type=str, default=None,
                        help="Path for output file with age calculations")

    args = parser.parse_args()

    # Create phecode instance
    phecode = Phecode(
        platform=args.platform,
        icd_file_path=args.icd_file_path,
        gbq_dataset_id=args.gbq_dataset_id
    )
    
    # Run add_age_at_first_event
    phecode.add_age_at_first_event(
        phecode_count_file_path=args.phecode_count_file_path,
        output_file_path=args.output_file_path
    )


def main_add_phecode_time_to_event():
    """Main entry point for add-phecode-time-to-event CLI command."""
    import argparse

    _enable_line_buffered_stdout()

    parser = argparse.ArgumentParser(
        description="Calculate time from study start to first phecode event for survival analysis"
    )
    
    # Required arguments
    parser.add_argument("--phecode_count_file_path", "-c", type=str, required=True,
                        help="Path to phecode counts CSV/TSV file")
    parser.add_argument("--cohort_file_path", "-f", type=str, required=True,
                        help="Path to cohort CSV/TSV file with study start dates")
    parser.add_argument("--study_start_date_col", "-s", type=str, required=True,
                        help="Column name containing study start dates")
    
    # Optional arguments
    parser.add_argument("--time_unit", type=str, default="days",
                        help="Time unit for calculations: 'days' or 'years' (default: days)")
    parser.add_argument("--output_file_path", "-o", type=str, default=None,
                        help="Path for output file with time calculations")
    
    args = parser.parse_args()
    
    # Run add_phecode_time_to_event (static method)
    Phecode.add_phecode_time_to_event(
        phecode_count_file_path=args.phecode_count_file_path,
        cohort_file_path=args.cohort_file_path,
        study_start_date_col=args.study_start_date_col,
        time_unit=args.time_unit,
        output_file_path=args.output_file_path
    )
