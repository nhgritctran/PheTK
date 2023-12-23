from tqdm import tqdm
from . import PheWAS
import numpy as np
import polars as pl
import os
import random


def generate_examples(phecode="GE_979.2", cohort_size=500, var_type="binary"):
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
    cols = {"person_id": np.array(range(1, cohort_size + 1)),
            "age": np.random.randint(18, 80, cohort_size),
            "sex": np.random.randint(0, 2, cohort_size),
            "pc1": np.random.uniform(-1, 1, cohort_size),
            "pc2": np.random.uniform(-1, 1, cohort_size),
            "pc3": np.random.uniform(-1, 1, cohort_size)}
    if var_type == "binary":
        cols["var_of_interest"] = np.random.randint(0, 2, cohort_size)
    elif var_type == "continuous":
        cols["var_of_interest"] = np.random.uniform(0.1, 10, cohort_size)

    cohort = pl.from_dict(cols)
    if var_type == "binary":
        case_ids = cohort.filter(pl.col("var_of_interest") == 1)["person_id"].unique().to_list()
        ctrl_ids = cohort.filter(pl.col("var_of_interest") == 0)["person_id"].unique().to_list()
    elif var_type == "continuous":
        var_mean = np.mean(cols.get("var_of_interest"))
        case_ids = cohort.filter(pl.col("var_of_interest") >= var_mean)["person_id"].unique().to_list()
        ctrl_ids = cohort.filter(pl.col("var_of_interest") < var_mean)["person_id"].unique().to_list()
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


def run():
    print("\033[1mHello, this is a demo of how to run PheWAS with PheTK.\033[0m")
    print()
    input("Press enter to continue...")
    print()
    print("\033[1mFirst, let's create some example data.\033[0m")
    print()
    print("We will create an example cohort with covariates age, sex, and 3 PCs.",
          "This data also contain our variable of interest which can be binary or continuous.",
          "In addition, we will also create an example phenotype profile data for this cohort.")
    var_type = input("Which data type would you like to use, binary or continuous?")
    if (var_type == "binary") or (var_type == "continuous"):
        generate_examples(var_type=var_type)
    else:
        var_type = input("Please enter either binary or continuous:")
        generate_examples(var_type=var_type)
    print()
    input("Press enter to continue...")
    print()
    print("Here is how the cohort data look like:")
    print(pl.read_csv("example_cohort.csv").head())
    print()
    input("Press enter to continue...")
    print()
    print("Here is how phenotype profile data look like:")
    print(pl.read_csv("example_phecode_counts.csv", dtypes={"phecode": str}).head())
    print()
    print("\033[1mNow we are ready to run PheWAS!\033[0m")
    print()
    print("For this demo, we can use the following command to run PheWAS in command line interface:")
    print("PheWAS --cohort_csv_path example_cohort.csv",
          "--phecode_count_csv_path example_phecode_counts.csv",
          "--phecode_version X",
          "--sex_at_birth_col sex",
          "--covariates age sex pc1 pc2 pc3",
          "--variable_of_interest var_of_interest",
          "--min_case 50",
          "--min_phecode_count 2",
          "--output_file_name example_phewas_results.csv")
    print()
    input("Press enter to run PheWAS!")
    print()
    phewas = PheWAS.PheWAS(cohort_csv_path="example_cohort.csv",
                           phecode_count_csv_path="example_phecode_counts.csv",
                           phecode_version="X",
                           sex_at_birth_col="sex",
                           covariate_cols=["age", "sex", "pc1", "pc2", "pc3"],
                           variable_of_interest="var_of_interest",
                           min_cases=50,
                           min_phecode_count=2,
                           output_file_name="example_phewas_results.csv")
    phewas.run()
    print()
    print("\033[1mThis is the end of the demo! Good luck!\033[0m")
    print()
