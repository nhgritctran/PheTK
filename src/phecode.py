from . import queries, utils
import os
import polars as pl


def count_phecode(db="aou", phecode_version="X"):
    """
    extract phecode counts for a biobank database
    :param db: defaults to "aou"; currently only option
    :param phecode_version: defaults to "X"; other option is "1.2"
    :return: phecode counts polars dataframe
    """

    # All of Us
    if db == "aou":
        # load phecode mapping file by version
        if phecode_version == "X":
            phecode_df = pl.read_csv("PyPheWAS/phecode/phecodeX.csv",
                                     dtypes={"phecode": str,
                                             "ICD": str})
        elif phecode_version == "1.2":
            phecode_df = pl.read_csv("PyPheWAS/phecode/phecode12.csv",
                                     dtypes={"phecode": str,
                                             "ICD": str,
                                             "phecode_unrolled": str})
        else:
            return "Invalid phecode version. Please choose either \"1.2\" or \"X\"."

        # generate query
        cdr = os.getenv("WORKSPACE_CDR")
        icd_query = queries.phecode_icd_query(cdr)

        # query data
        print("Start querying ICD codes...")
        icd_events = utils.polars_gbq(icd_query)
        print("Mapping ICD codes to phecodes...")
        if phecode_version == "X":
            phecode_counts = icd_events.join(phecode_df[["phecode", "ICD"]], how="inner", on="ICD")
        elif phecode_version == "1.2":
            phecode_counts = icd_events.join(phecode_df[["phecode_unrolled", "ICD"]], how="inner", on="ICD")
            phecode_counts = phecode_counts.rename({"phecode_unrolled": "phecode"})
        else:
            phecode_counts = None
        if not phecode_counts.is_empty() or phecode_counts is not None:
            phecode_counts = phecode_counts.drop("date").groupby(["person_id", "phecode"]).count()
    else:
        phecode_counts = None

    # report result
    if not phecode_counts.is_empty() or phecode_counts is not None:
        if db == "aou":
            db_val = "All of Us"
        else:
            db_val = None
        print(f"Successfully generated phecode {phecode_version} counts for {db_val} participants.")
        return phecode_counts
    else:
        print("No phecode count generated.")
        return None
