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
            print("\033[1mDone!")
        elif platform == "custom":
            if icd_df_path is not None:
                self.icd_events = pl.read_csv(icd_df_path,
                                              dtypes={"ICD": str})
            else:
                print("icd_df_path is required for custom database.")
                sys.exit(0)
        else:
            print("Invalid database. Parameter db only accepts \"aou\" (All of Us) or \"custom\".")
            sys.exit(0)

    def count_phecode(self, phecode_version="X", icd_version="US",
                      phecode_map_file_path=None, output_file_name=None):
        """
        Generate phecode counts from ICD counts
        :param phecode_version: defaults to "X"; other option is "1.2"
        :param icd_version: defaults to "US"; other option are "WHO" and "custom";
                            if "custom", user need to provide phecode_map_path
        :param phecode_map_file_path: path to custom phecode map table
        :param output_file_name: user specified output file name
        :return: phecode counts polars dataframe
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
        if "flag" not in icd_events.columns:
            icd_events = icd_events.with_columns(pl.when((pl.col("vocabulary_id") == "ICD9") |
                                                         (pl.col("vocabulary_id") == "ICD9CM"))
                                                 .then(9)
                                                 .when((pl.col("vocabulary_id") == "ICD10") |
                                                       (pl.col("vocabulary_id") == "ICD10M"))
                                                 .then(10)
                                                 .otherwise(0)
                                                 .alias("flag")
                                                 .cast(pl.Int8))
        else:
            icd_events = icd_events.with_columns(pl.col("flag").cast(pl.Int8))
        icd_events = icd_events[["person_id", "ICD", "flag"]]

        print()
        print(f"\033[1mMapping ICD codes to phecode {phecode_version}...")
        if phecode_version == "X":
            lazy_counts = icd_events.lazy().join(phecode_df.lazy(),
                                                 how="inner",
                                                 on=["ICD", "flag"])
        elif phecode_version == "1.2":
            lazy_counts = icd_events.lazy().join(phecode_df.lazy(),
                                                 how="inner",
                                                 on=["ICD", "flag"])
            lazy_counts = lazy_counts.rename({"phecode_unrolled": "phecode"})
        else:
            lazy_counts = None
        phecode_counts = lazy_counts.collect()
            
        if not phecode_counts.is_empty() or phecode_counts is not None:
            phecode_counts = phecode_counts.groupby(["person_id", "phecode"]).count()

        # report result
        if not phecode_counts.is_empty() or phecode_counts is not None:
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
            print("\033[1mNo phecode count generated.\033[0m")
            print()
