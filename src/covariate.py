import os
import polars as pl
import queries
import utils


def get_covariate(participant_ids,
                  natural_age=True,
                  age_at_last_event=True,
                  sex_at_birth=True,
                  ehr_length=True,
                  dx_code_count=True,
                  genetic_ancestry=False,
                  pc_count=0,
                  cdr_version=7):

    if is_instance(participant_ids, str):
        participant_ids = [participant_ids]

    if cdr_version == 7:
        cdr = os.getenv("WORKSPACE_CDR")
        if natural_age:
            natural_age_df = utils.polars_gbq(queries.natural_age_query(cdr))
        if age_at_last_event:
            pass
        if sex_at_birth:
            pass
        if genetic_ancestry:
            pass
        if pc_count > 0:
            pass

    else:
        print("Currently, only All of Us CDR v7 is supported.")
        return
