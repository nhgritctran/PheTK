from . import _queries, _utils
import os
import polars as pl
import sys


class Phecode:
    """
    Class Phecode extract ICD codes and map ICD codes to phecodes version 1.2 and X.
    Currently, supports ICD code extraction for All of Us OMOP data.
    For other databases, user is expected to provide an ICD code table for all participants in cohort of interest.
    """

    def __init__(self, db="aou", icd_df_path=None):
        """
        Instantiate based on parameter db
        :param db: supports:
            "aou": All of Us OMOP database
            "custom": other databases; icd_df must be not None if db = "custom"
        :param icd_df_path: path to ICD table csv file; required columns are "person_id", "ICD", and "vocabulary_id";
            "vocabulary_id" values should be "ICD9CM" or "ICD10CM"
        """
        self.db = db
        if db == "aou":
            self.cdr = os.getenv("WORKSPACE_CDR")
            self.icd_query = _queries.phecode_icd_query(self.cdr)
            print("\033[1mStart querying ICD codes...")
            self.icd_events = _utils.polars_gbq(self.icd_query)
            print("\033[1mDone!")
        elif db == "custom":
            self.icd_events = pl.read_csv(icd_df_path,
                                          dtypes={"ICD": str})
        else:
            print("Invalid database. Parameter db only accepts \"aou\" (All of Us) or \"custom\".")
            sys.exit(0)

    def count_phecode(self, phecode_version="X"):
        """
        Extract phecode counts for a biobank database
        :param phecode_version: defaults to "X"; other option is "1.2"
        :return: phecode counts polars dataframe
        """        
        # load phecode mapping file by version
        src_dir = os.path.dirname(__file__)
        phecode_mapping_file_path = os.path.join(src_dir, "..", "data", "phecode")
        if phecode_version == "X":
            phecode_mapping_file_path = os.path.join(phecode_mapping_file_path, "phecodeX.csv")
            # noinspection PyTypeChecker
            phecode_df = pl.read_csv(phecode_mapping_file_path,
                                     dtypes={"phecode": str,
                                             "ICD": str,
                                             "flag": pl.Int8,
                                             "code_val": float})
        elif phecode_version == "1.2":
            phecode_mapping_file_path = os.path.join(phecode_mapping_file_path, "phecode12.csv")
            # noinspection PyTypeChecker
            phecode_df = pl.read_csv(phecode_mapping_file_path,
                                     dtypes={"phecode": str,
                                             "ICD": str,
                                             "flag": pl.Int8,
                                             "exclude_range": str,
                                             "phecode_unrolled": str})
        else:
            print("Unsupported phecode version. Supports phecode \"1.2\" and \"X\".")
            sys.exit(0)

        # make a copy of self.icd_events
        icd_events = self.icd_events.clone()
        icd_events = icd_events.with_columns(pl.when(pl.col("vocabulary_id") == "ICD9CM")
                                             .then(9)
                                             .otherwise(10)
                                             .alias("flag")
                                             .cast(pl.Int8))
        icd_events = icd_events.drop(["date", "vocabulary_id"])

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
            if self.db == "aou":
                db_val = "All of Us"
            elif self.db == "custom":
                db_val = "custom"
            else:
                db_val = None
            file_name = self.db + "_phecode" + phecode_version.upper().replace(".", "") + "_counts.csv"
            phecode_counts.write_csv(file_name)
            print(f"\033[1mSuccessfully generated phecode {phecode_version} counts for {db_val} participants!\n"
                  f"\033[1mSaved to {file_name}!\033[0m")
            print()

        else:
            print("\033[1mNo phecode count generated.\033[0m")
            print()
