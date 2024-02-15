def phecode_icd_query(ds):
    """
    This method is exclusively for All of Us platform
    :param ds: Google BigQuery dataset ID containing OMOP data tables
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
                    c.vocabulary_id AS vocabulary_id,
                    c.concept_code AS ICD
                FROM
                    {ds}.condition_occurrence AS co
                INNER JOIN
                    {ds}.concept AS c
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
                    c.vocabulary_id AS vocabulary_id,
                    c.concept_code AS ICD
                FROM
                    {ds}.condition_occurrence AS co
                INNER JOIN
                    {ds}.concept AS c
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
                    c.vocabulary_id AS vocabulary_id,
                    c.concept_code AS ICD
                FROM
                    {ds}.observation AS o
                INNER JOIN
                    {ds}.concept as c
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
                    c.vocabulary_id AS vocabulary_id,
                    c.concept_code AS ICD
                FROM
                    {ds}.observation AS o
                INNER JOIN
                    {ds}.concept as c
                ON
                    o.observation_source_concept_id = c.concept_id
                WHERE
                    c.vocabulary_id in ("ICD9CM", "ICD10CM")
                )
            )
    """

    return query


def natural_age_query(ds, participant_ids):
    """
    This method is exclusively for All of Us platform
    :param ds: Google BigQuery dataset ID containing OMOP data tables
    :param participant_ids: list of participant IDs to query
    :return: a SQL query that would generate a table
            contains participant IDs and their natural age
    """
    query: str = f"""
        SELECT
            DISTINCT p.person_id, 
            DATETIME_DIFF(
                IF(DATETIME(death_datetime) IS NULL, CURRENT_DATETIME(), DATETIME(death_datetime)), 
                DATETIME(birth_datetime), 
                DAY
            )/365.2425 AS natural_age
        FROM
            {ds}.person AS p
        LEFT JOIN
            {ds}.death AS d
        ON
            p.person_id = d.person_id
        WHERE
            p.person_id IN {participant_ids}
    """

    return query


def ehr_dx_code_query(ds, participant_ids):
    """
    This method is exclusively for All of Us platform.
    In condition occurrence table, diagnosis codes belongs to ICD9CM, ICD10CM and SNOMED, are counted.
    In observation table, diagnosis codes belongs to ICD9CM and ICD10CM are counted.
    :param ds: Google BigQuery dataset ID containing OMOP data tables
    :param participant_ids: list of participant IDs to query
    :return: a SQL query that would generate a table contains participant IDs and
            their ehr length (days), diagnosis code count(ICD & SNOMED), and age at last event
    """
    query: str = f"""
        SELECT DISTINCT
            df1.person_id,
            (DATETIME_DIFF(MAX(date), MIN(date), DAY) + 1) AS ehr_length,
            COUNT(code) AS dx_code_occurrence_count,
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
                    {ds}.condition_occurrence AS co
                INNER JOIN
                    {ds}.concept AS c
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
                    {ds}.condition_occurrence AS co
                INNER JOIN
                    {ds}.concept AS c
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
                    {ds}.observation AS o
                INNER JOIN
                    {ds}.concept AS c
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
                    {ds}.observation AS o
                INNER JOIN
                    {ds}.concept AS c
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
                    {ds}.person
                WHERE
                    person_id IN {participant_ids}
            ) AS df2
        ON
            df1.person_id = df2.person_id
        GROUP BY 
            df1.person_id
    """

    return query


def sex_at_birth(ds, participant_ids):
    """
    This method is exclusively for All of Us platform
    :param ds: Google BigQuery dataset ID containing OMOP data tables
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
                    {ds}.person
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
                    {ds}.person
                WHERE
                    sex_at_birth_source_concept_id = 1585847
                AND
                    person_id IN {participant_ids}
                )
            )
    """

    return query
