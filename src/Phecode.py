from . import _queries, _utils
import os
import polars as pl
import sys


class Phecode:

    def __init__(self, db="aou"):
        if db == "aou":
            self.db = db
        else:
            print("Invalid database. Currently, only \"aou\" (All of Us) is supported.")
            sys.exit(0)
        self.cdr = os.getenv("WORKSPACE_CDR")
        self.icd_query = _queries.phecode_icd_query(self.cdr)
        print("\033[1mStart querying ICD codes...")
        self.icd_events = _utils.polars_gbq(self.icd_query)
        print("\033[1mDone!")

    def count_phecode(self, phecode_version="X"):
        """
        Extract phecode counts for a biobank database
        :param phecode_version: defaults to "X"; other option is "1.2"
        :return: phecode counts polars dataframe
        """        
        # load phecode mapping file by version
        if phecode_version.upper() == "X":
            # noinspection PyTypeChecker
            phecode_df = pl.read_csv("PyPheWAS/phecode/phecodeX.csv",
                                     dtypes={"phecode": str,
                                             "ICD": str,
                                             "phecode_top": str,
                                             "code_val": float})
        elif phecode_version.upper() == "1.2":
            # noinspection PyTypeChecker
            phecode_df = pl.read_csv("PyPheWAS/phecode/phecode12.csv",
                                     dtypes={"phecode": str,
                                             "ICD": str,
                                             "phecode_unrolled": str,
                                             "exclude_range": str})
        else:
            return "Invalid phecode version. Please choose either \"1.2\" or \"X\"."

        # make a copy of self.icd_events
        icd_events = self.icd_events.clone()

        print()
        print(f"\033[1mMapping ICD codes to phecode {phecode_version}...")
        if phecode_version == "X":
            phecode_counts = icd_events.join(phecode_df[["phecode", "ICD"]], how="inner", on="ICD")
        elif phecode_version == "1.2":
            phecode_counts = icd_events.join(phecode_df[["phecode_unrolled", "ICD"]], how="inner", on="ICD")
            phecode_counts = phecode_counts.rename({"phecode_unrolled": "phecode"})
        else:
            phecode_counts = None
        if not phecode_counts.is_empty() or phecode_counts is not None:
            phecode_counts = phecode_counts.drop(["date", "vocabulary_id"]).groupby(["person_id", "phecode"]).count()

        # report result
        if not phecode_counts.is_empty() or phecode_counts is not None:
            if self.db == "aou":
                db_val = "All of Us"
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
