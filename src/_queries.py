def phecode_icd_query(cdr):
    """
    This method is exclusively for All of Us platform
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
                    co.person_id,
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
                    co.person_id,
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
                    o.person_id,
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
                    o.person_id,
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


def natural_age_query(cdr, participant_ids):
    """
    This method is exclusively for All of Us platform
    :param cdr: All of Us Curated Data Repository
    :param participant_ids: list of participant IDs to query
    :return: a SQL query that would generate a table
            contains participant IDs and their natural age
    """
    query: str = f"""
        SELECT
            p.person_id, 
            DATETIME_DIFF(MIN(CURRENT_DATETIME(), death_date), DATETIME(birth_datetime), DAY)/365.2425 AS natural_age
        FROM
            {cdr}.person AS p
        LEFT JOIN
            {cdr}.death AS d
        ON
            p.person_id = d.person_id
        WHERE
            person_id IN {participant_ids}
    """

    return query


def ehr_dx_code_count_query(cdr, participant_ids):
    """
    This method is exclusively for All of Us platform.
    In condition occurrence table, diagnosis codes belongs to ICD9CM, ICD10CM and SNOMED, are counted.
    In observation table, diagnosis codes belongs to ICD9CM and ICD10CM are counted.
    :param cdr: All of Us Curated Data Repository
    :param participant_ids: list of participant IDs to query
    :return: a SQL query that would generate a table contains participant IDs and
            their ehr length (days), diagnosis code count(ICD & SNOMED), and age at last event
    """
    query: str = f"""
        SELECT DISTINCT
            df1.person_id,
            DATETIME_DIFF(MAX(date), MIN(date), DAY) AS ehr_length,
            COUNT(code) AS dx_code_occurence_count,
            COUNT(DISTINCT(code)) AS dx_condition_count,
            DATETIME_DIFF(MAX(date), MIN(birthday), DAY)/365.2425 AS age_at_last_event,
        FROM
            (
                (
                SELECT DISTINCT
                    co.person_id,
                    co.condition_start_date AS date,
                    c.concept_code AS code
                FROM
                    {cdr}.condition_occurrence AS co
                INNER JOIN
                    {cdr}.concept AS c
                ON
                    co.condition_source_value = c.concept_code
                WHERE
                    c.vocabulary_id IN ("ICD9CM", "ICD10CM", "SNOMED")
                    AND
                    person_id IN {participant_ids}
                )
            UNION DISTINCT
                (
                SELECT DISTINCT
                    co.person_id,
                    co.condition_start_date AS date,
                    c.concept_code AS code
                FROM
                    {cdr}.condition_occurrence AS co
                INNER JOIN
                    {cdr}.concept AS c
                ON
                    co.condition_source_concept_id = c.concept_id
                WHERE
                    c.vocabulary_id IN ("ICD9CM", "ICD10CM", "SNOMED")
                    AND
                    person_id IN {participant_ids}
                )
            UNION DISTINCT
                (
                SELECT DISTINCT
                    o.person_id,
                    o.observation_date AS date,
                    c.concept_code AS code
                FROM
                    {cdr}.observation AS o
                INNER JOIN
                    {cdr}.concept AS c
                ON
                    o.observation_source_value = c.concept_code
                WHERE
                    c.vocabulary_id IN ("ICD9CM", "ICD10CM")
                    AND
                    person_id IN {participant_ids}
                )
            UNION DISTINCT
                (
                SELECT DISTINCT
                    o.person_id,
                    o.observation_date AS date,
                    c.concept_code AS code
                FROM
                    {cdr}.observation AS o
                INNER JOIN
                    {cdr}.concept AS c
                ON
                    o.observation_source_concept_id = c.concept_id
                WHERE
                    c.vocabulary_id IN ("ICD9CM", "ICD10CM")
                    AND
                    person_id IN {participant_ids}
                )
            ) AS df1
        INNER JOIN
            (
                SELECT
                    person_id, 
                    EXTRACT(DATE FROM DATETIME(birth_datetime)) AS birthday
                FROM
                    {cdr}.person
                WHERE
                    person_id IN {participant_ids}
            ) AS df2
        ON
            df1.person_id = df2.person_id
        GROUP BY 
            df1.person_id
    """

    return query


def sex_at_birth(cdr, participant_ids):
    """
    This method is exclusively for All of Us platform
    :param cdr: All of Us Curated Data Repository
    :param participant_ids: list of participant IDs to query
    :return: a SQL query that would generate a table contains participant IDs and
            their sex at birth; male = 1, female = 0
    """
    query: str = f"""
        SELECT
            *
        FROM
            (
                (
                SELECT
                    person_id,
                    1 AS sex_at_birth
                FROM
                    {cdr}.person
                WHERE
                    sex_at_birth_source_concept_id = 1585846
                AND
                    person_id IN {participant_ids}
                )
            UNION DISTINCT
                (
                SELECT
                    person_id,
                    0 AS sex_at_birth
                FROM
                    {cdr}.person
                WHERE
                    sex_at_birth_source_concept_id = 1585847
                AND
                    person_id IN {participant_ids}
                )
            UNION DISTINCT
                (
                SELECT
                    person_id,
                    1 AS sex_at_birth
                FROM
                    {cdr}.observation
                WHERE
                    observation_source_concept_id = 8507
                AND
                    person_id IN {participant_ids}
                )
            UNION DISTINCT
                (
                SELECT
                    person_id,
                    0 AS sex_at_birth
                FROM
                    {cdr}.observation
                WHERE
                    observation_source_concept_id = 8532
                AND
                    person_id IN {participant_ids}
                )
            )
    """

    return query
