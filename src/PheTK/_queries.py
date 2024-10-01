def phecode_icd_query(ds):
    """
    This method is optimized for All of Us platform.

    It includes 3 queries: icd_query, v_icd_vocab_query, and final_query.
    icd_query retrieves all ICD codes from OMOP database.
    v_icd_vocab_query get the ICD codes starting with "V" from icd_query and check vocabulary_id using concept_id.
    final_query union distinct icd_query without V codes
    and v_icd_vocab_query which has V codes with proper vocabulary_ids.

    The reason for this is to ensure vocabulary_id values of V codes, many of which overlap between ICD9CM & ICD10CM,
    are correct.

    :param ds: Google BigQuery dataset ID containing OMOP data tables
    :return: a SQL query that would generate a table contains participant IDs and their ICD codes from unique dates
    """
    icd_query: str = f"""
        (
            SELECT DISTINCT
                co.person_id,
                co.condition_start_date AS date,
                c.vocabulary_id AS vocabulary_id,
                c.concept_code AS ICD,
                co.condition_concept_id AS concept_id
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
                c.concept_code AS ICD,
                co.condition_concept_id AS concept_id
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
                c.concept_code AS ICD,
                o.observation_concept_id AS concept_id
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
                c.concept_code AS ICD,
                o.observation_concept_id AS concept_id
            FROM
                {ds}.observation AS o
            INNER JOIN
                {ds}.concept as c
            ON
                o.observation_source_concept_id = c.concept_id
            WHERE
                c.vocabulary_id in ("ICD9CM", "ICD10CM")
        )
    """

    v_icd_vocab_query: str = f"""
        SELECT DISTINCT
            v_icds.person_id,
            v_icds.date,
            v_icds.ICD,
            c.vocabulary_id
        FROM
            (
                SELECT
                    *
                FROM
                    ({icd_query}) AS icd_events
                WHERE
                    icd_events.ICD LIKE "V%"
            ) AS v_icds
        INNER JOIN
            {ds}.concept_relationship AS cr
        ON
            v_icds.concept_id = cr.concept_id_1
        INNER JOIN
            {ds}.concept AS c
        ON
            cr.concept_id_2 = c.concept_id
        WHERE
            c.vocabulary_id IN ("ICD9CM", "ICD10CM")
        AND
            v_icds.ICD = c.concept_code
        AND NOT
            v_icds.vocabulary_id != c.vocabulary_id
    """

    final_query: str = f"""
        (
            SELECT DISTINCT
                person_id,
                date,
                ICD,
                vocabulary_id
            FROM 
                ({icd_query})
            WHERE
                NOT ICD LIKE "V%"
        )
        UNION DISTINCT
        (
            SELECT DISTINCT
                *
            FROM
                ({v_icd_vocab_query})
        )
    """

    return final_query


def date_of_birth_query(ds):
    """
    This method is used to get date of birth of participants from OMOP data
    :param ds: Google BigQuery dataset ID containing OMOP data tables
    :return: a SQL query to get date of birth of all participants
    """
    query: str = f"""
        SELECT DISTINCT 
            person_id,
            EXTRACT(DATE FROM DATETIME(birth_datetime)) AS date_of_birth
        FROM
            {ds}.person
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
    In condition occurrence table, diagnosis codes belongs to ICD9CM, ICD10CM, are counted.
    In observation table, diagnosis codes belongs to ICD9CM and ICD10CM are counted.
    :param ds: Google BigQuery dataset ID containing OMOP data tables
    :param participant_ids: list of participant IDs to query
    :return: a SQL query that would generate a table contains participant IDs and
            their ehr length (days), diagnosis code count(ICD), and age at last event
    """
    query: str = f"""
        SELECT DISTINCT
            df1.person_id,
            MAX(date) AS last_ehr_date,
            (DATETIME_DIFF(MAX(date), MIN(date), DAY) + 1)/365.2425 AS ehr_length,
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
                    c.vocabulary_id IN ("ICD9CM", "ICD10CM")
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
