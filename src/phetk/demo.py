# noinspection PyUnresolvedReferences
from phetk.phewas import PheWAS
from phetk._utils import generate_mock_phewas_data
import polars as pl
import sys


def _prompt():
    """
    Display interactive prompt for demo progression with quit option.

    Pauses execution to allow user to read information and provides
    option to exit demo at any point by typing "quit".

    Returns:
        Continues execution or exits program based on user input.
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
        verbose=False,
        method="logit"):
    """
    Execute interactive PheWAS demonstration with mock data generation and analysis.

    Guides users through complete PheWAS workflow including data generation,
    parameter configuration, analysis execution, and results interpretation
    using synthetic datasets.

    Args:
        covariates_cols: Column names to use as covariates in PheWAS analysis.
        independent_variable_of_interest: Name of primary variable for analysis.
        phecode_to_process: Specific phecodes to analyze or "all" for complete analysis.
        verbose: Whether to display detailed progress information during analysis.
        method: Regression method ("logit", "cox", "firth_logit", or "firth_cox").

    Returns:
        Completes demonstration workflow and displays results.
    """
    valid_methods = ("logit", "cox", "firth_logit", "firth_cox")
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
            generate_mock_phewas_data(var_type=var_type, data_has_both_sexes=data_has_both_sexes)
    _prompt()
    print("\033[1mCohort Data Preview (500 participants):\033[0m")
    print(pl.read_csv("example_cohort.tsv", separator="\t").head())
    _prompt()
    print("\033[1mPhecode Count Data Preview:\033[0m")
    print(pl.read_csv("example_phecode_counts.tsv", separator="\t", schema_overrides={"phecode": str}).head())
    print()
    print("This shows phecodes mapped from each person's ICD codes and their counts.")
    _prompt()
    print("\033[1mStep 2: Choose Regression Method\033[0m")
    print()
    print("Available methods:")
    print("  - logit       : Standard logistic regression")
    print("  - cox         : Cox proportional hazards regression")
    print("  - firth_logit : Firth penalized logistic regression")
    print("  - firth_cox   : Firth penalized Cox regression")
    print()
    method_input = input(f"Which regression method? ({'/'.join(valid_methods)}) [{method}] ")
    if method_input.strip():
        if method_input.lower() == "quit":
            print()
            print("\033[1mExiting demo. Thank you!\033[0m")
            sys.exit(0)
        elif method_input.lower() in valid_methods:
            method = method_input.lower()
        else:
            print(f"Invalid method '{method_input}'. Using default: {method}")
    _prompt()
    print("\033[1mStep 3: Run PheWAS Analysis\033[0m")
    print()
    print("Command line equivalent:")
    print("\033[1mpython3 -m phetk.phewas --cohort_file_path\033[0m example_cohort.tsv",
          "\033[1m--phecode_count_file_path\033[0m example_phecode_counts.tsv",
          "\033[1m--phecode_version\033[0m X",
          "\033[1m--sex_at_birth_col\033[0m sex",
          "\033[1m--covariates\033[0m age sex pc1 pc2 pc3",
          "\033[1m--independent_variable_of_interest\033[0m independent_variable_of_interest",
          f"\033[1m--method\033[0m {method}",
          "\033[1m--min_cases\033[0m 50",
          "\033[1m--min_phecode_count\033[0m 2",
          "\033[1m--output_file_path\033[0m example_phewas_results.tsv")
    print()
    input("\033[1mPress enter to run PheWAS...\033[0m")
    print()
    if isinstance(covariates_cols, tuple):
        covariates_cols = list(covariates_cols)

    phewas_kwargs = {
        "cohort_file_path": "example_cohort.tsv",
        "phecode_count_file_path": "example_phecode_counts.tsv",
        "phecode_version": "X",
        "sex_at_birth_col": "sex",
        "phecode_to_process": phecode_to_process,
        "covariate_cols": list(covariates_cols),
        "independent_variable_of_interest": independent_variable_of_interest,
        "min_cases": 50,
        "min_phecode_count": 2,
        "output_file_path": "example_phewas_results.tsv",
        "verbose": verbose,
        "method": method,
    }
    # Add Cox params if method requires time-to-event
    if method in ("cox", "firth_cox"):
        phewas_kwargs["cox_control_observed_time_col"] = "observed_time"
        phewas_kwargs["cox_phecode_observed_time_col"] = "phecode_observed_time"

    phewas = PheWAS(**phewas_kwargs)
    phewas.run()
    print("\033[1mTop Results (sorted by p-value):\033[0m")
    print(pl.read_csv("example_phewas_results.tsv", separator="\t", schema_overrides={"phecode": str}).sort(by="p_value").head())
    print()
    method_desc = {
        "logit": "logistic regressions",
        "cox": "Cox proportional hazards regressions",
        "firth_logit": "Firth penalized logistic regressions",
        "firth_cox": "Firth penalized Cox regressions",
    }
    print(f"PheWAS ran {method_desc.get(method, method)}: phecode ~ independent_variable + covariates")
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
