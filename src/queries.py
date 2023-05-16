def phecode_icd_query(cdr):
    """
    this method is exclusively for All of Us platform
    :param cdr: All of Us Curated Data Repository
    :return: a SQL query that would generate a table contains participant IDs and their ICD codes from unique dates
    """
    query: str = f"""
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

    return query


def ehr_code_query(cdr):
    """
    this method is exclusively for All of Us platform
    :param cdr: All of Us Curated Data Repository
    :return: a SQL query that would generate a table
             contains participant IDs and their ICD/SNOMED codes from unique dates
    """
    query: str = f"""
        SELECT DISTINCT
            *
        FROM
            (
                (
                SELECT DISTINCT
                    CAST(co.person_id AS STRING) AS person_id,
                    co.condition_start_date AS date,
                    c.concept_code AS code
                FROM
                    {cdr}.condition_occurrence AS co
                INNER JOIN
                    {cdr}.concept AS c
                ON
                    co.condition_source_value = c.concept_code
                WHERE
                    c.vocabulary_id in ("ICD9CM", "ICD10CM", "SNOMED")
                )
            UNION DISTINCT
                (
                SELECT DISTINCT
                    CAST(co.person_id AS STRING) AS person_id,
                    co.condition_start_date AS date,
                    c.concept_code AS code
                FROM
                    {cdr}.condition_occurrence AS co
                INNER JOIN
                    {cdr}.concept AS c
                ON
                    co.condition_source_concept_id = c.concept_id
                WHERE
                    c.vocabulary_id in ("ICD9CM", "ICD10CM", "SNOMED")
                )
            UNION DISTINCT
                (
                SELECT DISTINCT
                    CAST(o.person_id AS STRING) AS person_id,
                    o.observation_date AS date,
                    c.concept_code AS code
                FROM
                    {cdr}.observation AS o
                INNER JOIN
                    {cdr}.concept as c
                ON
                    o.observation_source_value = c.concept_code
                WHERE
                    c.vocabulary_id in ("ICD9CM", "ICD10CM", "SNOMED")
                )
            UNION DISTINCT
                (
                SELECT DISTINCT
                    CAST(o.person_id AS STRING) AS person_id,
                    o.observation_date AS date,
                    c.concept_code AS code
                FROM
                    {cdr}.observation AS o
                INNER JOIN
                    {cdr}.concept as c
                ON
                    o.observation_source_concept_id = c.concept_id
                WHERE
                    c.vocabulary_id in ("ICD9CM", "ICD10CM", "SNOMED")
                )
            )
        """

    return query
