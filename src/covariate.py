from . import _paths, _queries, _utils
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.notebook import tqdm
import os
import pandas as pd
import polars as pl


def _get_ancestry_preds(db_version, user_project, participant_ids):
    """
    this method specifically designed for All of Us database
    :param db_version: version of database; supports All of Us CDR v6 & v7
    :param user_project: proxy of GOOGLE_PROJECT environment variable of current workspace in All of Us workbench
    :param participant_ids: participant IDs of interest
    :return: ancestry_preds data of specific version as polars dataframe object
    """
    if db_version == 7:
        ancestry_preds = pd.read_csv(_paths.cdr7_ancestry_pred_path,
                                     sep="\t",
                                     storage_options={"requester_pays": True,
                                                      "user_project": user_project})
        ancestry_preds = pl.from_pandas(ancestry_preds)
        ancestry_preds = ancestry_preds.with_columns(pl.col("pca_features").str.replace(r"\[", "")) \
            .with_columns(pl.col("pca_features").str.replace(r"\]", "")) \
            .with_columns(pl.col("pca_features").str.split(",").arr.get(i).alias(f"pc{i}") for i in range(16)) \
            .with_columns(pl.col(f"pc{i}").str.replace(" ", "").cast(float) for i in range(16)) \
            .drop(["probabilities", "pca_features", "ancestry_pred_other"]) \
            .rename({"research_id": "person_id",
                     "ancestry_pred": "genetic_ancestry"})\
            .filter(pl.col("person_id").is_in(participant_ids))
    else:
        ancestry_preds = None

    return ancestry_preds


def _get_covariates(participant_ids,
                    natural_age=True,
                    age_at_last_event=True,
                    sex_at_birth=True,
                    ehr_length=True,
                    dx_code_count=True,
                    genetic_ancestry=False,
                    first_n_pcs=0,
                    db_version=7):
    """
    this method specifically designed for All of Us database
    core internal function to generate covariate data for a set of participant IDs
    :param participant_ids: IDs of interest
    :param natural_age: age of participants as of today
    :param age_at_last_event: age of participants at their last diagnosis event in EHR record
    :param sex_at_birth: sex at birth from survey and observation
    :param ehr_length: number of days that EHR record spans
    :param dx_code_count: count of diagnosis codes, including ICD9CM, ICD10CM & SNOMED
    :param genetic_ancestry: predicted ancestry based on sequencing data
    :param first_n_pcs: number of first principle components to include
    :param db_version: version of database; supports All of Us version 7
    :return: polars dataframe object
    """

    # initial data prep
    if isinstance(participant_ids, str) or isinstance(participant_ids, int):
        participant_ids = (participant_ids, )
    elif isinstance(participant_ids, list):
        participant_ids = tuple(participant_ids)
    df = pl.DataFrame({"person_id": participant_ids})

    # All of Us CDR v7
    if db_version == 7:
        cdr = os.getenv("WORKSPACE_CDR")
        user_project = os.getenv("GOOGLE_PROJECT")
        participant_ids = tuple([int(i) for i in participant_ids])

        if natural_age:
            natural_age_df = _utils.polars_gbq(_queries.natural_age_query(cdr, participant_ids))
            df = df.join(natural_age_df, how="left", on="person_id")
            # print("Retrieved natural age...")

        if age_at_last_event or ehr_length or dx_code_count:
            temp_df = _utils.polars_gbq(_queries.ehr_dx_code_query(cdr, participant_ids))
            cols_to_keep = ["person_id"]
            if age_at_last_event:
                # print("Retrieved age at last event...")
                cols_to_keep.append("age_at_last_event")
            if ehr_length:
                # print("Retrieved ehr length...")
                cols_to_keep.append("ehr_length")
            if dx_code_count:
                # print("Retrieved diagnosis code count...")
                cols_to_keep.append("dx_code_count")
            df = df.join(temp_df[cols_to_keep], how="left", on="person_id")

        if sex_at_birth:
            sex_df = _utils.polars_gbq(_queries.sex_at_birth(cdr, participant_ids))
            df = df.join(sex_df, how="left", on="person_id")

        if genetic_ancestry or first_n_pcs > 0:
            temp_df = _get_ancestry_preds(db_version, user_project, participant_ids)
            cols_to_keep = ["person_id"]
            if genetic_ancestry:
                # print("Retrieved genetic ancestry...")
                cols_to_keep.append("genetic_ancestry")
            if first_n_pcs > 0:
                # print(f"Retrieved first {first_n_pcs} PCs...")
                cols_to_keep = cols_to_keep + [f"pc{i}" for i in range(first_n_pcs)]
            df = df.join(temp_df[cols_to_keep], how="left", on="person_id")

        return df

    else:
        print("Currently, only All of Us CDR v7 is supported.")
        return


def get_covariates(participant_ids,
                   natural_age=True,
                   age_at_last_event=True,
                   sex_at_birth=True,
                   ehr_length=True,
                   dx_code_count=True,
                   genetic_ancestry=False,
                   first_n_pcs=0,
                   db_version=7,
                   chunk_size=10000):
    """
    multithreading version of _get_covariates method passing 10,000 IDs to each thread
    :param participant_ids: IDs of interest
    :param natural_age: age of participants as of today
    :param age_at_last_event: age of participants at their last diagnosis event in EHR record
    :param sex_at_birth: sex at birth from survey and observation
    :param ehr_length: number of days that EHR record spans
    :param dx_code_count: count of diagnosis codes, including ICD9CM, ICD10CM & SNOMED
    :param genetic_ancestry: predicted ancestry based on sequencing data
    :param first_n_pcs: number of first principle components to include
    :param db_version: version of database; supports All of Us version 7
    :param chunk_size: defaults to 10,000; number of IDs per thread
    :return: csv file and polars dataframe object
    """
    chunks = [
        list(participant_ids)[i*chunk_size:(i+1)*chunk_size] for i in range((len(participant_ids)//chunk_size)+1)
    ]
    with ThreadPoolExecutor() as executor:
        jobs = [
            executor.submit(
                _get_covariates,
                chunk,
                natural_age,
                age_at_last_event,
                sex_at_birth,
                ehr_length,
                dx_code_count,
                genetic_ancestry,
                first_n_pcs,
                db_version
            ) for chunk in chunks
        ]
        result_list = [job.result() for job in tqdm(as_completed(jobs), total=len(chunks))]

    result_list = [result for result in result_list if result is not None]
    df = result_list[0]
    for i in range(1, len(chunks)):
        df = pl.concat([df, result_list[i]])
    df = df.unique()
    df.write_csv("covariates.csv")

    return df
