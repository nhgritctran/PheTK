from google.cloud import bigquery
import polars as pl
import pyarrow as pa


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
