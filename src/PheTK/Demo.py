# noinspection PyUnresolvedReferences
from PheTK.PheWAS import PheWAS
from tqdm import tqdm
import numpy as np
import polars as pl
import os
import random
import sys


def generate_examples(phecode="GE_979.2", cohort_size=500, var_type="binary",
                      data_has_both_sexes=True):
    # load phecode mapping file to get all phecodes
    phetk_dir = os.path.dirname(__file__)
    phecode_mapping_file_path = os.path.join(phetk_dir, "phecode")
    phecode_mapping_file_path = os.path.join(phecode_mapping_file_path, "phecodeX.csv")
    # noinspection PyTypeChecker
    phecode_df = pl.read_csv(phecode_mapping_file_path,
                             dtypes={"phecode": str,
                                     "ICD": str,
                                     "flag": pl.Int8,
                                     "code_val": float})
    phecodes = phecode_df["phecode"].unique().to_list()
    phecodes.remove(phecode)  # exclude target phecode for background data

    # mock cohort
    if data_has_both_sexes:
        n_sex = 2
    else:
        n_sex = 1
    cols = {"person_id": np.array(range(1, cohort_size + 1)),
            "age": np.random.randint(18, 80, cohort_size),
            "sex": np.random.randint(0, n_sex, cohort_size),
            "pc1": np.random.uniform(-1, 1, cohort_size),
            "pc2": np.random.uniform(-1, 1, cohort_size),
            "pc3": np.random.uniform(-1, 1, cohort_size)}
    if var_type == "binary":
        cols["independent_variable_of_interest"] = np.random.randint(0, 2, cohort_size)
    elif var_type == "continuous":
        cols["independent_variable_of_interest"] = np.random.uniform(0.1, 10, cohort_size)

    cohort = pl.from_dict(cols)
    if var_type == "binary":
        case_ids = cohort.filter(pl.col("independent_variable_of_interest") == 1)["person_id"].unique().to_list()
        ctrl_ids = cohort.filter(pl.col("independent_variable_of_interest") == 0)["person_id"].unique().to_list()
    elif var_type == "continuous":
        var_mean = np.mean(cols.get("independent_variable_of_interest"))
        case_ids = cohort.filter(pl.col("independent_variable_of_interest") >= var_mean)["person_id"].unique().to_list()
        ctrl_ids = cohort.filter(pl.col("independent_variable_of_interest") < var_mean)["person_id"].unique().to_list()
    else:
        print("Error: var_type can only be \"binary\" or \"continuous\".")
        return

    # mock phecode_counts
    phecode_counts = None
    for i in tqdm(range(cohort_size)):
        if i == 0:
            cols = {}
            # noinspection PyTypeChecker
            ids = random.sample(
                case_ids, np.random.randint(round(len(case_ids) * 0.5), len(case_ids) * 0.8)
            ) + ctrl_ids[:round(len(ctrl_ids) / 10)]
            cols["person_id"] = np.array(ids)
            cols["phecode"] = np.array([phecode] * len(ids))
            cols["count"] = np.random.randint(1, 3, len(ids))
        else:
            cols = {}
            ids = random.sample(cohort["person_id"].to_list(),
                                np.random.randint(round(cohort_size * 0.5), round(cohort_size * 0.9)))
            cols["person_id"] = np.array(ids)
            cols["phecode"] = np.array([phecodes[np.random.randint(1, len(phecodes))]] * len(ids))
            cols["count"] = np.random.randint(1, 10, len(ids))
        df = pl.from_dict(cols)
        if phecode_counts is None:
            phecode_counts = df
        else:
            phecode_counts = pl.concat([phecode_counts, df])
    phecode_counts = phecode_counts.unique(["person_id", "phecode"]).sort(by="person_id")

    # save data
    cohort.write_csv("example_cohort.csv")
    phecode_counts.write_csv("example_phecode_counts.csv")
    print(
        "Generated data saved to \033[1m\"example_cohort.csv\"\033[0m",
        "and \033[1m\"example_phecode_counts.csv\"\033[0m"
    )


def _prompt():
    print()
    answer = input("Press enter to continue...")
    if answer.lower() == "quit":
        print()
        print("\033[1mGood luck!\033[0m")
        sys.exit(0)
    print()


def run(covariates_cols=("age", "sex", "pc1", "pc2", "pc3"),
        independent_variable_of_interest="independent_variable_of_interest",
        phecode_to_process="all",
        verbose=False):
    print("\033[1mHello and welcome to PheTK PheWAS demo.\033[0m")
    print("This is a quick demonstration to introduce a basic PheWAS analysis using mock data.",
          "It should take less than 1 minute running without pauses.",
          "For a detailed tutorial, please check out the included jupyter notebooks.",
          "Enter \"quit\" in any prompt to quit.")
    _prompt()
    print("\033[1mFirst, let's create some example data.\033[0m")
    print()
    print(f"We will create an example cohort with covariates {covariates_cols}.",
          "This data also contains our variable of interest which can be binary or continuous.")
    print()
    print("In addition, we will also create an example phenotype profile data for this cohort.",
          "This table contains all phecodes mapped from ICD codes from each person's EHR and their counts.")
    print()
    var_type = input("Which data type would you like the variable of interest to be? (binary/continuous) ")
    while (var_type.lower() != "binary") and (var_type.lower() != "continuous") and (var_type.lower() != "quit"):
        var_type = input("Please enter either binary or continuous:")
    if var_type.lower() == "quit":
        print()
        print("\033[1mGood luck!\033[0m")
    else:
        data_has_both_sexes = input("Would you like data to have both sexes? (yes/no) ")
        while (data_has_both_sexes.lower() != "yes") \
                and (data_has_both_sexes.lower() != "no") \
                and (data_has_both_sexes.lower() != "quit"):
            data_has_both_sexes = input("Please enter either yes or no:")
        if data_has_both_sexes.lower() == "quit":
            print()
            print("\033[1mGood luck!\033[0m")
        else:
            if data_has_both_sexes.lower() == "yes":
                data_has_both_sexes = True
            else:
                data_has_both_sexes = False
            generate_examples(var_type=var_type, data_has_both_sexes=data_has_both_sexes)
    _prompt()
    print("\033[1mWe created a cohort of 500 people and here is how the cohort data look like:\033[0m")
    print(pl.read_csv("example_cohort.csv").head())
    _prompt()
    print("\033[1mHere is how the phecode count data look like:\033[0m")
    print(pl.read_csv("example_phecode_counts.csv", dtypes={"phecode": str}).head())
    _prompt()
    print("\033[1mNow we are ready to run PheWAS!\033[0m")
    print()
    print("If run in command line interface, the analysis below can be run with the following command:")
    print("\033[1mpython3 -m PheTK.PheWAS --cohort_csv_path\033[0m example_cohort.csv",
          "\033[1m--phecode_count_csv_path\033[0m example_phecode_counts.csv",
          "\033[1m--phecode_version\033[0m X",
          "\033[1m--sex_at_birth_col\033[0m sex",
          "\033[1m--covariates\033[0m age sex pc1 pc2 pc3",
          "\033[1m--independent_variable_of_interest\033[0m independent_variable_of_interest",
          "\033[1m--min_case\033[0m 50",
          "\033[1m--min_phecode_count\033[0m 2",
          "\033[1m--output_file_name\033[0m example_phewas_results.csv")
    print()
    input("\033[1mPress enter to run PheWAS!\033[0m")
    print()
    if isinstance(covariates_cols, tuple):
        covariates_cols = list(covariates_cols)
    phewas = PheWAS(cohort_csv_path="example_cohort.csv",
                    phecode_count_csv_path="example_phecode_counts.csv",
                    phecode_version="X",
                    sex_at_birth_col="sex",
                    phecode_to_process=phecode_to_process,
                    covariate_cols=list(covariates_cols),
                    independent_variable_of_interest=independent_variable_of_interest,
                    min_cases=50,
                    min_phecode_count=2,
                    output_file_name="example_phewas_results.csv",
                    verbose=verbose)
    phewas.run()
    print("\033[1mHere is how example_phewas_results.csv look like:\033[0m")
    if independent_variable_of_interest == "independent_variable_of_interest" and phecode_to_process == "all":
        print("In this example, we intentionally generated data with Cystic Fibrosis as a significant hit.")
    print(pl.read_csv("example_phewas_results.csv", dtypes={"phecode": str}).sort(by="p_value").head())
    print()
    print("\033[1mThis is the end of the demo!\033[0m")
    print()
    print("\033[1mGood luck!\033[0m")


if __name__ == "__main__":
    run()
