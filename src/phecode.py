from google.cloud import bigquery
import os
import polars as pl


def _polars_gbq(query):
    """
    take a SQL query and return result as polars dataframe
    :param query: BigQuery SQL query
    :return: polars dataframe
    """
    client = bigquery.Client()
    query_job = client.query(query)
    rows = query_job.result()
    df = pl.from_arrow(rows.to_arrow())

    return df


def phecode_count(db="aou", phecode_version="X"):
    """
    extract phecode counts for a biobank database
    :param db: defaults to "aou"; currently only option
    :param phecode_version: defaults to "X"; currently only option
    :return: phecode counts polars dataframe
    """

    # All of Us
    if db == "aou":
        # load phecode mapping file by version
        if phecode_version == "X":
            phecode_df = pl.read_csv("../phecode/phecodeX.csv")
        else:
            phecode_df = None

        cdr = os.getenv("WORKSPACE_CDR")
        icd_query = f"""
        SELECT DISTINCT
            *
        FROM
            (
                (
                SELECT DISTINCT
                    CAST(co.person_id AS STRING) AS person_id,
                    co.condition_start_date AS date,
                    c.concept_code AS ICD
                FROM
                    {cdr}.condition_occurrence AS co
                INNER JOIN
                    {cdr}.concept AS c
                ON
                    co.condition_source_value = c.concept_code
                WHERE
                    c.vocabulary_id in ("ICD9CM", "ICD10CM")
                )
            UNION DISTINCT
                (
                SELECT DISTINCT
                    CAST(co.person_id AS STRING) AS person_id,
                    co.condition_start_date AS date,
                    c.concept_code AS ICD
                FROM
                    {cdr}.condition_occurrence AS co
                INNER JOIN
                    {cdr}.concept AS c
                ON
                    co.condition_source_concept_id = c.concept_id
                WHERE
                    c.vocabulary_id in ("ICD9CM", "ICD10CM")
                )
            UNION DISTINCT
                (
                SELECT DISTINCT
                    CAST(o.person_id AS STRING) AS person_id,
                    o.observation_date AS date,
                    c.concept_code AS ICD
                FROM
                    {cdr}.observation AS o
                INNER JOIN
                    {cdr}.concept as c
                ON
                    o.observation_source_value = c.concept_code
                WHERE
                    c.vocabulary_id in ("ICD9CM", "ICD10CM")
                )
            UNION DISTINCT
                (
                SELECT DISTINCT
                    CAST(o.person_id AS STRING) AS person_id,
                    o.observation_date AS date,
                    c.concept_code AS ICD
                FROM
                    {cdr}.observation AS o
                INNER JOIN
                    {cdr}.concept as c
                ON
                    o.observation_source_concept_id = c.concept_id
                WHERE
                    c.vocabulary_id in ("ICD9CM", "ICD10CM")
                )
            )
        """
        icd_events = _polars_gbq(icd_query)
        phecode_counts = icd_events.join(phecode_df[["phecode", "ICD"]], how="inner", on="ICD")
        phecode_counts = phecode_counts.drop("date").groupby(["person_id", "phecode"]).count()
    else:
        phecode_counts = None

    # report result
    if phecode_counts:
        if db == "aou":
            db_val = "All of Us"
        else:
            db_val = None
        print(f"Successfully generated phecode {phecode_version} counts for {db_val}.")
        return phecode_counts
    else:
        print("No phecode count generated.")
        return None
