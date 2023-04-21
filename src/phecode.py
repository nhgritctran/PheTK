from google.cloud import bigquery
import os
import polars as pl


def extraction(db="aou"):

    if db == "aou":

        cdr = os.getenv("WORKSPACE_CDR")

        query = f"""
        SELECT
            person_id, icd_code
        FROM
            (
                SELECT
                    co.person_id, co.condition_source_value AS icd_code
                FROM
                    {cdr}.condition_occurrence AS co
                INNER JOIN
                    {cdr}.concept as c
                ON
                    co.condition_source_value = c.concept_code
                WHERE
                    c.vocabulary_id in ("ICD9CM", "ICD10CM")
            )
            
            UNION
            
            (
                SELECT
                    co.person_id, co.condition_source_value AS icd_code
                FROM
                    {cdr}.condition_occurrence AS co
                INNER JOIN
                    {cdr}.concept as c
                ON
                    co.condition_source_value = c.concept_code
                WHERE
                    c.vocabulary_id in ("ICD9CM", "ICD10CM")
            )
        """

    else:
        pass
