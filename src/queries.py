phecode_icd_query = f"""
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

ehr_code_query = f"""
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
