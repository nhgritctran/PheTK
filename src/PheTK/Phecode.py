import os
import polars as pl
import sys
# noinspection PyUnresolvedReferences,PyProtectedMember
from PheTK import _queries, _utils


class Phecode:
    """
    Class Phecode extract ICD codes and map ICD codes to phecodes version 1.2 and X.
    Currently, supports ICD code extraction for All of Us OMOP data.
    For other databases, user is expected to provide an ICD code table for all participants in cohort of interest.
    """

    def __init__(self, platform="aou", icd_df_path=None):
        """
        Instantiate based on parameter db
        :param platform: supports:
            "aou": All of Us OMOP database
            "custom": other databases; icd_df must be not None if db = "custom"
        :param icd_df_path: path to ICD table csv file; required columns are "person_id", "ICD", and "vocabulary_id";
            "vocabulary_id" values should be "ICD9CM" or "ICD10CM"
        """
        self.platform = platform

        if platform == "aou":
            self.cdr = os.getenv("WORKSPACE_CDR")
            self.icd_query = _queries.phecode_icd_query(self.cdr)
            print("\033[1mStart querying ICD codes...")
            self.icd_events = _utils.polars_gbq(self.icd_query)

        elif platform == "custom":
            if icd_df_path is not None:
                print("\033[1mLoading user's ICD data from file...")
                self.icd_events = pl.read_csv(icd_df_path,
                                              dtypes={"ICD": str})
            else:
                print("icd_df_path is required for custom platform.")
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

        print("\033[1mDone!")

    def count_phecode(self, phecode_version="X", icd_version="US",
                      phecode_map_file_path=None, output_file_name=None):
        """
        Generate phecode counts from ICD counts
        :param phecode_version: defaults to "X"; other option is "1.2"
        :param icd_version: defaults to "US"; other option are "WHO" and "custom";
                            if "custom", user need to provide phecode_map_path
        :param phecode_map_file_path: path to custom phecode map table
        :param output_file_name: user specified output file name
        :return: phecode counts csv file
        """        
        # load phecode mapping file by version or by custom path
        phecode_df = _utils.get_phecode_mapping_table(
            phecode_version=phecode_version,
            icd_version=icd_version,
            phecode_map_file_path=phecode_map_file_path,
            keep_all_columns=False
        )

        # make a copy of self.icd_events
        icd_events = self.icd_events.clone()

        # keep only necessary columns
        icd_events = icd_events[["person_id", "date", "ICD", "flag"]]

        print()
        print(f"\033[1mMapping ICD codes to phecode {phecode_version}...")
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
                file_name = "{0}_{1}_phecode{2}_counts.csv".format(self.platform, icd_version,
                                                                   phecode_version.upper().replace(".", ""))
            else:
                file_name = output_file_name
            phecode_counts.write_csv(file_name)
            print(f"\033[1mSuccessfully generated phecode {phecode_version} counts for cohort participants!\n"
                  f"\033[1mSaved to {file_name}!\033[0m")
            print()

        else:
            print("\033[1mNo phecode count generated. Check your input data.\033[0m")
            print()

    def add_age_at_first_event(self, phecode_count_file_path):
        """
        Calculate age at first event based on input date at first event and birthdays from OMOP data
        :param phecode_count_file_path: path to phecode count csv file; must have columns "person_id", "phecode",
            and "first_event_date"
        :return: new phecode counts csv file with age at first event
        """
        phecode_counts = pl.read_csv(phecode_count_file_path,
                                     dtypes={"phecode": str,
                                             "first_event_date": pl.Date()})
        print("\033[1mPhecode counts loaded.")

        print("\033[1mQuerying date of birth.")
        date_of_birth_df = _utils.polars_gbq(_queries.date_of_birth_query(self.cdr))

        print("\033[1mCalculating age at first event.")
        phecode_counts = phecode_counts.join(date_of_birth_df, how="inner", on=["person_id"])
        phecode_counts = phecode_counts.with_columns(
            (pl.col("first_event_date") - pl.col("date_of_birth")).dt.total_days()/365.2425
        ).alias("age_at_first_event")

        phecode_counts.write_csv("phecode_counts_with_event_age.csv")
        print("\033[1mDone! Saved as phecode_counts_with_event_age.csv.")
