import os
import polars as pl
import sys
# noinspection PyUnresolvedReferences,PyProtectedMember
from phetk import _queries, _utils


class Phecode:
    """
    Class Phecode extract ICD codes and map ICD codes to phecodes version 1.2 and X.
    Currently, supports ICD code extraction for All of Us OMOP data.
    For other databases, the user is expected to provide an ICD code table for all participants in the cohort of interest.
    """

    def __init__(self, platform="aou", icd_file_path=None):
        """
        Instantiate based on parameter db
        :param platform: supports:
            "aou": All of Us OMOP database
            "custom": other databases; icd_df must be not None if db = "custom"
        :param icd_file_path: path to the ICD table csv/tsv file; required columns are "person_id", "ICD", and "vocabulary_id";
            "vocabulary_id" values should be "ICD9CM" or "ICD10CM"
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

    def count_phecode(self, phecode_version="X", icd_version="US",
                      phecode_map_file_path=None, output_file_name=None):
        """
        Generate phecode counts from ICD counts
        :param phecode_version: defaults to "X"; other option is "1.2"
        :param icd_version: defaults to "US"; other options are "WHO" and "custom";
                            if "custom", users need to provide phecode_map_path
        :param phecode_map_file_path: path to custom phecode map table
        :param output_file_name: user-specified output file name
        :return: phecode counts tsv file
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
            if output_file_name is None:
                file_name = "{0}_{1}_phecode{2}_counts.tsv".format(self.platform, icd_version,
                                                                   phecode_version.upper().replace(".", ""))
            else:
                file_name = output_file_name
            phecode_counts.write_csv(file_name, separator="\t")
            print(f"Successfully generated phecode{phecode_version} counts for cohort participants!")
            print()
            print(f"Saved to\033[1m {file_name}\033[0m")
            print()

        else:
            print("\033[1mNo phecode count generated. Check your input data.\033[0m")
            print()

    def add_age_at_first_event(self, phecode_count_file_path):
        """
        Calculate age at the first event based on input date at the first event and birthdays from OMOP data
        :param phecode_count_file_path: the path to the phecode count /tsv file; must have columns "person_id", "phecode",
            and "first_event_date"
        :return: new phecode counts tsv file with age at the first event
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
            query_function=_queries.natural_age_query,
            ds=self.cdr,
            id_list=participant_ids,
            chunk_size=1000
        )

        date_of_birth_df = _utils.polars_gbq_chunk(
            query_list=query_list,
        )
        # # old version without chunking
        # date_of_birth_df = _utils.polars_gbq(_queries.natural_age_query(ds=self.cdr, participant_ids=participant_ids))

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

        phecode_counts.write_csv("phecode_counts_with_event_age.tsv", separator="\t")
        print("Done!")
        print()
        print(f"Saved to\033[1m phecode_counts_with_event_age.tsv\033[0m. "
              f"Age at first event column name is {col_name}.")
        print()

    @staticmethod
    def add_phecode_time_to_event(
            phecode_count_file_path,
            cohort_file_path,
            study_start_date_col,
            time_unit="days"
    ):
        """
        Calculate time to event for each phecode, based on the study start date of each participant in the study cohort
        :param phecode_count_file_path: path to phecode count csv/tsv file
        :param cohort_file_path: path to cohort csv/tsv file
        :param study_start_date_col: column name of study start date
        :param time_unit: unit of time to calculate phecode time, defaults to "days", accepts "days" or "years"
        :return: new phecode counts tsv file with time to event
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

        phecode_counts.write_csv("phecode_counts_with_phecode_time_to_event.tsv", separator="\t")
        print("Done!")
        print()
        print(f"Saved to\033[1m phecode_counts_with_phecode_time_to_event.tsv\033[0m. "
              f"Phecode time to event column name is {col_name}.")
        print()
