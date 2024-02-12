from google.cloud import bigquery
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
    :param phecode_version: defaults to "X"; other option is "1.2"
    :param icd_version: defaults to "US"; other option are "WHO" and "custom";
                        if "custom", user need to provide phecode_map_path
    :param phecode_map_file_path: path to custom phecode map table
    :param keep_all_columns: defaults to True
    :return: phecode mapping table as polars dataframe
    """
    # load phecode mapping file by version or by custom path
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
                sys.exit(0)
        else:
            print("Invalid icd_version. Available icd_version values are US, WHO and custom.")
            sys.exit(0)
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
            sys.exit(0)
        elif icd_version == "custom":
            if phecode_map_file_path is None:
                print("Please provide phecode_map_path for custom icd_version")
                sys.exit(0)
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
        sys.exit(0)

    return phecode_df
