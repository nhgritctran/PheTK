# noinspection PyUnresolvedReferences
from phetk.phewas import PheWAS
from tqdm import tqdm
import numpy as np
import polars as pl
import os
import random
import sys


def generate_examples(phecode="GE_979.2", cohort_size=500, var_type="binary",
                      data_has_both_sexes=True):
    """
    Generate mock cohort and phecode count data for PheWAS demonstration.
    
    Creates synthetic datasets with specified characteristics including target
    phecode enrichment in cases, realistic phecode distributions, and customizable
    variable types for testing PheWAS functionality.
    
    :param phecode: Target phecode to enrich in cases for demonstration.
    :type phecode: str
    :param cohort_size: Number of participants in generated cohort.
    :type cohort_size: int
    :param var_type: Type of independent variable ("binary" or "continuous").
    :type var_type: str
    :param data_has_both_sexes: Whether to include both sexes in generated data.
    :type data_has_both_sexes: bool
    :return: Saves example_cohort.tsv and example_phecode_counts.tsv files.
    :rtype: None
    """
    # load the phecode mapping file to get all phecodes
    phetk_dir = os.path.dirname(__file__)
    phecode_mapping_file_path = os.path.join(phetk_dir, "phecode")
    phecode_mapping_file_path = os.path.join(phecode_mapping_file_path, "phecodeX.csv")
    # noinspection PyTypeChecker
    phecode_df = pl.read_csv(phecode_mapping_file_path,
                             schema_overrides={"phecode": str,
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
    cohort.write_csv("example_cohort.tsv", separator="\t")
    phecode_counts.write_csv("example_phecode_counts.tsv", separator="\t")
    print("Generated data saved to:")
    print("  - \033[1mexample_cohort.tsv\033[0m")
    print("  - \033[1mexample_phecode_counts.tsv\033[0m")


def _prompt():
    """
    Display interactive prompt for demo progression with quit option.
    
    Pauses execution to allow user to read information and provides
    option to exit demo at any point by typing "quit".
    
    :return: Continues execution or exits program based on user input.
    :rtype: None
    """
    print()
    answer = input("Press Enter to continue (or 'quit' to exit)...")
    if answer.lower() == "quit":
        print()
        print("\033[1mExiting demo. Thank you!\033[0m")
        sys.exit(0)
    print()


def run(covariates_cols=("age", "sex", "pc1", "pc2", "pc3"),
        independent_variable_of_interest="independent_variable_of_interest",
        phecode_to_process=None,
        verbose=False):
    """
    Execute interactive PheWAS demonstration with mock data generation and analysis.
    
    Guides users through complete PheWAS workflow including data generation,
    parameter configuration, analysis execution, and results interpretation
    using synthetic datasets.
    
    :param covariates_cols: Column names to use as covariates in PheWAS analysis.
    :type covariates_cols: tuple[str, ...]
    :param independent_variable_of_interest: Name of primary variable for analysis.
    :type independent_variable_of_interest: str
    :param phecode_to_process: Specific phecodes to analyze or "all" for complete analysis.
    :type phecode_to_process: str
    :param verbose: Whether to display detailed progress information during analysis.
    :type verbose: bool
    :return: Completes demonstration workflow and displays results.
    :rtype: None
    """
    print("\033[1mWelcome to PheTK PheWAS Demo\033[0m")
    print("This quick demo shows PheWAS analysis using mock data (~1 minute).")
    print("For detailed tutorials, see the included documentations.")
    print("Type 'quit' at any prompt to exit.")
    _prompt()
    print("\033[1mStep 1: Generate Example Data\033[0m")
    print()
    print(f"Creating cohort with covariates: {covariates_cols}")
    print("The variable of interest can be:")
    print("  - Binary (e.g., genotype present/absent)")
    print("  - Continuous (e.g., lab measurements)")
    print()
    print("Also creating phecode counts from ICD codes in EHR data.")
    print()
    var_type = input("Which data type would you like the variable of interest to be? (binary/continuous) ")
    while (var_type.lower() != "binary") and (var_type.lower() != "continuous") and (var_type.lower() != "quit"):
        var_type = input("Please enter either binary or continuous: ")
    if var_type.lower() == "quit":
        print()
        print("\033[1mExiting demo. Thank you!\033[0m")
    else:
        data_has_both_sexes = input("Would you like data to have both sexes? (yes/no) ")
        while (data_has_both_sexes.lower() != "yes") \
                and (data_has_both_sexes.lower() != "no") \
                and (data_has_both_sexes.lower() != "quit"):
            data_has_both_sexes = input("Please enter either yes or no: ")
        if data_has_both_sexes.lower() == "quit":
            print()
            print("\033[1mExiting demo. Thank you!\033[0m")
        else:
            if data_has_both_sexes.lower() == "yes":
                data_has_both_sexes = True
            else:
                data_has_both_sexes = False
            generate_examples(var_type=var_type, data_has_both_sexes=data_has_both_sexes)
    _prompt()
    print("\033[1mCohort Data Preview (500 participants):\033[0m")
    print(pl.read_csv("example_cohort.tsv", separator="\t").head())
    _prompt()
    print("\033[1mPhecode Count Data Preview:\033[0m")
    print(pl.read_csv("example_phecode_counts.tsv", separator="\t", schema_overrides={"phecode": str}).head())
    print()
    print("This shows phecodes mapped from each person's ICD codes and their counts.")
    _prompt()
    print("\033[1mStep 2: Run PheWAS Analysis\033[0m")
    print()
    print("Command line equivalent:")
    print("\033[1mpython3 -m phetk.phewas --cohort_file_path\033[0m example_cohort.tsv",
          "\033[1m--phecode_count_file_path\033[0m example_phecode_counts.tsv",
          "\033[1m--phecode_version\033[0m X",
          "\033[1m--sex_at_birth_col\033[0m sex",
          "\033[1m--covariates\033[0m age sex pc1 pc2 pc3",
          "\033[1m--independent_variable_of_interest\033[0m independent_variable_of_interest",
          "\033[1m--min_cases\033[0m 50",
          "\033[1m--min_phecode_count\033[0m 2",
          "\033[1m--output_file_path\033[0m example_phewas_results.tsv")
    print()
    input("\033[1mPress enter to run PheWAS...\033[0m")
    print()
    if isinstance(covariates_cols, tuple):
        covariates_cols = list(covariates_cols)
    phewas = PheWAS(
        cohort_file_path="example_cohort.tsv",
        phecode_count_file_path="example_phecode_counts.tsv",
        phecode_version="X",
        sex_at_birth_col="sex",
        phecode_to_process=phecode_to_process,
        covariate_cols=list(covariates_cols),
        independent_variable_of_interest=independent_variable_of_interest,
        min_cases=50,
        min_phecode_count=2,
        output_file_path="example_phewas_results.tsv",
        verbose=verbose)
    phewas.run()
    print("\033[1mTop Results (sorted by p-value):\033[0m")
    print(pl.read_csv("example_phewas_results.tsv", separator="\t", schema_overrides={"phecode": str}).sort(by="p_value").head())
    print()
    print("PheWAS ran logistic regressions: phecode ~ independent_variable + covariates")
    print()
    print("Key parameters:")
    print(f"  - min_phecode_count: {2} (minimum count to be a case)")
    print(f"  - min_cases: {50} (minimum cases/controls to run regression)")
    print()
    print("Each phecode was tested only if sufficient cases and controls were available.")
    print()
    print("Note: Cystic Fibrosis (GE_979.2) was enriched in the data as an example hit.")
    print()
    print("\033[1mDemo Complete!\033[0m")


if __name__ == "__main__":
    run()
