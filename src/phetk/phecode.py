import os
import polars as pl
import sys
# noinspection PyUnresolvedReferences,PyProtectedMember
from phetk import _queries, _utils


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
            icd_file_path: str | None = None
    ):
        """
        Initialize Phecode object for ICD code extraction and phecode mapping.
        
        Sets up data source configuration and loads ICD code data either from
        All of Us OMOP database or custom file. Validates vocabulary types and
        creates ICD version flags for downstream processing.
        
        :param platform: Data platform, supports "aou" (All of Us) or "custom".
        :type platform: str
        :param icd_file_path: Path to ICD table CSV/TSV file with columns "person_id", "ICD", "vocabulary_id".
        :type icd_file_path: str | None
        """
        self.platform = platform

        if platform == "aou":
            self.cdr = os.getenv("WORKSPACE_CDR")
            self.icd_query = _queries.phecode_icd_query(self.cdr)
            print("Start querying ICD codes...")
            self.icd_events = _utils.polars_gbq(self.icd_query)

        elif platform == "custom":
            if icd_file_path is not None:
                print("Loading user's ICD data from file...")
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

        print("Done!")

    def count_phecode(
            self,
            phecode_version: str = "X",
            icd_version: str = "US",
            phecode_map_file_path: str | None = None,
            output_file_path: str | None = None
    ) -> None:
        """
        Generate phecode counts from ICD code data.
        
        Maps ICD codes to phecodes using specified version and mapping table,
        aggregates counts per person-phecode combination, and calculates first
        event dates. Creates output file with person-level phecode statistics.
        
        :param phecode_version: Phecode version to use, "X" or "1.2".
        :type phecode_version: str
        :param icd_version: ICD mapping version, "US", "WHO", or "custom".
        :type icd_version: str
        :param phecode_map_file_path: Path to custom phecode mapping table.
        :type phecode_map_file_path: str | None
        :param output_file_path: Path for output TSV file.
        :type output_file_path: str | None
        :return: Creates phecode counts TSV file with person_id, phecode, count, and first_event_date.
        :rtype: None
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

        # make a copy of self.icd_events
        icd_events = self.icd_events.clone()

        # keep only the necessary columns
        icd_events = icd_events[["person_id", "date", "ICD", "flag"]]

        print()
        print(f"Mapping ICD codes to phecode{phecode_version_string}...")
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

        # report result
        if not phecode_counts.is_empty():
            if output_file_path is None:
                file_path = "{0}_{1}_phecode{2}_counts.tsv".format(self.platform, icd_version,
                                                                   phecode_version.upper().replace(".", ""))
            else:
                file_path = output_file_path
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
        
        :param phecode_count_file_path: Path to phecode counts TSV file with person_id, phecode, and first_event_date columns.
        :type phecode_count_file_path: str
        :param output_file_path: Path for output file with age calculations.
        :type output_file_path: str | None
        :return: Creates enhanced phecode counts TSV file with age_at_first_event column.
        :rtype: None
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
        
        :param phecode_count_file_path: Path to phecode counts CSV/TSV file.
        :type phecode_count_file_path: str
        :param cohort_file_path: Path to cohort CSV/TSV file with study start dates.
        :type cohort_file_path: str
        :param study_start_date_col: Column name containing study start dates.
        :type study_start_date_col: str
        :param time_unit: Time unit for calculations, "days" or "years".
        :type time_unit: str
        :param output_file_path: Path for output file with time calculations.
        :type output_file_path: str | None
        :return: Creates enhanced phecode counts TSV file with phecode_time_to_event column.
        :rtype: None
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
    
    parser = argparse.ArgumentParser(
        description="Generate phecode counts from ICD code data"
    )
    
    # Platform arguments
    parser.add_argument("--platform", type=str, default="aou",
                        help="Data platform: 'aou' or 'custom' (default: aou)")
    parser.add_argument("--icd_file_path", type=str, default=None,
                        help="Path to ICD table CSV/TSV file (required for custom platform)")
    
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
    
    args = parser.parse_args()
    
    # Create phecode instance
    phecode = Phecode(
        platform=args.platform,
        icd_file_path=args.icd_file_path
    )
    
    # Run count_phecode
    phecode.count_phecode(
        phecode_version=args.phecode_version,
        icd_version=args.icd_version,
        phecode_map_file_path=args.phecode_map_file_path,
        output_file_path=args.output_file_path
    )


def main_add_age_at_first_event():
    """Main entry point for add-age-at-first-event CLI command."""
    import argparse
    
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
    
    # Output argument
    parser.add_argument("--output_file_path", "-o", type=str, default=None,
                        help="Path for output file with age calculations")
    
    args = parser.parse_args()
    
    # Create phecode instance
    phecode = Phecode(
        platform=args.platform,
        icd_file_path=args.icd_file_path
    )
    
    # Run add_age_at_first_event
    phecode.add_age_at_first_event(
        phecode_count_file_path=args.phecode_count_file_path,
        output_file_path=args.output_file_path
    )


def main_add_phecode_time_to_event():
    """Main entry point for add-phecode-time-to-event CLI command."""
    import argparse
    
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
