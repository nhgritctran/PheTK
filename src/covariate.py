import os
import paths
import pandas as pd
import polars as pl
import queries
import utils


def get_covariates(participant_ids,
                   natural_age=True,
                   age_at_last_event=True,
                   sex_at_birth=True,
                   ehr_length=True,
                   dx_code_count=True,
                   genetic_ancestry=False,
                   pc_count=0,
                   cdr_version=7):

    # initial data prep
    if isinstance(participant_ids, str):
        participant_ids = (participant_ids, )
    elif isinstance(participant_ids, list):
        participant_ids = tuple(participant_ids)
    df = pl.DataFrame(participant_ids, schema={"person_id": str})

    # All of Us CDR v7
    if cdr_version == 7:
        cdr = os.getenv("WORKSPACE_CDR")
        project = os.getenv("GOOGLE_PROJECT")
        ancestry_pred_path = paths.cdr7_ancestry_pred_path

        if natural_age:
            natural_age_df = utils.polars_gbq(queries.natural_age_query(cdr, participant_ids))
            df = df.join(natural_age_df, how="left", on="person_id")

        if age_at_last_event or ehr_length or dx_code_count:
            temp_df = utils.polars_gbq(queries.ehr_dx_code_query(cdr, participant_ids))
            cols_to_keep = []
            if age_at_last_event:
                cols_to_keep.append("age_at_last_event")
            if ehr_length:
                cols_to_keep.append("ehr_length")
            if dx_code_count:
                cols_to_keep.append("dx_code_count")
            df = df.join(temp_df[cols_to_keep], how="left", on="person_id")

        if sex_at_birth:
            sex_df = utils.polars_gbq(queries.sex_at_birth(cdr, participant_ids))
            df = df.join(sex_df, how="left", on="person_id")

        if genetic_ancestry or pc_count > 0:
            ancestry_preds = pd.read_csv(ancestry_pred_path, sep="\t", storage_options={"requester_pays": True,
                                                                                        "user_projects": project})
            ancestry_preds = pl.from_pandas(ancestry_preds)
            pass

        return df

    else:
        print("Currently, only All of Us CDR v7 is supported.")
        return
