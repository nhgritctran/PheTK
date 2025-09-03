from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime
from io import StringIO
from lifelines import CoxPHFitter, utils as u
from multiprocessing import get_context
from tqdm import tqdm
import argparse
import copy
import numpy as np
import os
import pandas as pd
import polars as pl
import statsmodels
import statsmodels.api as sm
import sys
import warnings
# noinspection PyUnresolvedReferences,PyProtectedMember
from phetk import _utils



class PheWAS:

    def __init__(
        self,
        phecode_version: str,
        phecode_count_file_path: str,
        cohort_file_path: str,
        covariate_cols: list[str] | str,
        independent_variable_of_interest: str,
        sex_at_birth_col: str,
        male_as_one: bool = True,
        cox_start_date_col: str | None = None,
        cox_control_observed_time_col: str | None = None,
        cox_phecode_observed_time_col: str | None = None,
        cox_stratification_col: str | None = None,
        cox_fallback_step_size: float = 0.1,
        icd_version: str = "US",
        phecode_map_file_path: str | None = None,
        phecode_to_process: list[str] | str | None = None,
        min_cases: int = 50,
        min_phecode_count: int = 2,
        use_exclusion: bool = False,
        output_file_path: str | None = None,
        verbose: bool = False,
        suppress_warnings: bool = True,
        method: str = "logit",
        batch_size: int | None = None,
        fall_back_to_serial: bool = False
    ):
        """
        Initialize PheWAS analysis object with configuration parameters and input data.
        
        Performs data validation, loads phecode mapping tables, processes input files,
        validates column compatibility, and sets up analysis parameters for either
        logistic or Cox regression analysis.
        
        :param phecode_version: Phecode version to use ("1.2" or "X").
        :type phecode_version: str
        :param phecode_count_file_path: Path to CSV/TSV file containing phecode counts for participants.
        :type phecode_count_file_path: str
        :param cohort_file_path: Path to CSV/TSV file containing cohort data with covariates.
        :type cohort_file_path: str
        :param covariate_cols: Column names to use as covariates, excluding independent variable.
        :type covariate_cols: list[str] | str
        :param independent_variable_of_interest: Name of primary independent variable column.
        :type independent_variable_of_interest: str
        :param sex_at_birth_col: Name of sex/gender column, should contain 0/1 values.
        :type sex_at_birth_col: str
        :param male_as_one: If True, male=1 and female=0; if False, male=0 and female=1.
        :type male_as_one: bool
        :param cox_start_date_col: Column containing study start dates for Cox regression, optional for Cox.
        Date to exclude participants with pre-existing phenotype from cases of a particular phecode.
        :type cox_start_date_col: str | None
        :param cox_control_observed_time_col: Column containing censoring time for controls in Cox regression.
        :type cox_control_observed_time_col: str | None
        :param cox_phecode_observed_time_col: Column containing time to event for cases in Cox regression.
        :type cox_phecode_observed_time_col: str | None
        :param cox_stratification_col: Column name for stratification in Cox regression.
        :type cox_stratification_col: str | None
        :param cox_fallback_step_size: Fallback step size when Cox regression fails to converge, default 0.1.
        :type cox_fallback_step_size: float
        :param icd_version: ICD version ("US", "WHO", or "custom").
        :type icd_version: str
        :param phecode_map_file_path: Path to custom phecode mapping file, required if icd_version="custom".
        :type phecode_map_file_path: str | None
        :param phecode_to_process: Specific phecodes to analyze, None for all available.
        :type phecode_to_process: list[str] | str | None
        :param min_cases: Minimum number of cases required to test a phecode.
        :type min_cases: int
        :param min_phecode_count: Minimum phecode count to qualify as a case.
        :type min_phecode_count: int
        :param use_exclusion: Whether to use phecode exclusion ranges, only for phecode 1.2.
        :type use_exclusion: bool
        :param output_file_path: Output file path, auto-generated if None.
        :type output_file_path: str | None
        :param verbose: If True, print progress information for each phecode.
        :type verbose: bool
        :param suppress_warnings: If True, suppress convergence and statistical warnings.
        :type suppress_warnings: bool
        :param method: Analysis method ("logit" for logistic regression or "cox" for Cox regression).
        :type method: str
        :param batch_size: Number of phecodes to process per batch for parallelization. If None, defaults to 1 for logit and 10 for cox.
        :type batch_size: int | None
        :param fall_back_to_serial: Whether to fall back to serial processing when parallelization fails.
        :type fall_back_to_serial: bool
        """

        # For dsub
        self.phecode_version = phecode_version
        self.phecode_count_file_path = phecode_count_file_path
        self.cohort_file_path = cohort_file_path
        self.input_covariate_cols = covariate_cols
        self.male_as_one = male_as_one
        self.icd_version = icd_version
        self.phecode_map_file_path = phecode_map_file_path
        self.phecode_to_process = phecode_to_process
        self.dsub = None

        # Even when running with dsub, the instantiation steps below will still be run as a good check for input issue(s)
        _utils.print_banner("Creating PheWAS Instance")
        print()

        # for debugging
        if verbose:
            print(f"DEBUG: cohort_file_path = {self.cohort_file_path}")
            print(f"DEBUG: phecode_count_file_path = {self.phecode_count_file_path}")
            print(f"DEBUG: phecode_map_file_path = {self.phecode_map_file_path}")
            print(f"DSUB DEBUG: COHORT_FILE_PATH env var = {os.getenv('COHORT_FILE_PATH', 'NOT_SET')}")
            print(f"DSUB DEBUG: PHECODE_COUNT_FILE_PATH env var = {os.getenv('PHECODE_COUNT_FILE_PATH', 'NOT_SET')}")
            print(f"DSUB DEBUG: PHECODE_MAP_FILE_PATH env var = {os.getenv('PHECODE_MAP_FILE_PATH', 'NOT_SET')}")
            try:
                print("DSUB DEBUG: Files in /mnt/data/input/:")
                for root, dirs, files in os.walk('/mnt/data/input/'):
                    level = root.replace('/mnt/data/input/', '').count(os.sep)
                    indent = ' ' * 2 * (level + 1)
                    print(f"{indent}{os.path.basename(root)}/")
                    sub_indent = ' ' * 2 * (level + 2)
                    for file in files:
                        print(f"{sub_indent}{file}")
            except Exception as e:
                print(f"DSUB DEBUG: Could not list /mnt/data/input/: {e}")
            print()

        # Load the phecode mapping file by version or by custom path
        self.phecode_df = _utils.get_phecode_mapping_table(
            phecode_version=phecode_version,
            icd_version=icd_version,
            phecode_map_file_path=phecode_map_file_path,
            keep_all_columns=True
        )

        # Load phecode counts data for all participants
        # noinspection PyTypeChecker
        phecode_sep = _utils.detect_delimiter(phecode_count_file_path)
        self.phecode_counts = pl.read_csv(
            phecode_count_file_path,
            separator=phecode_sep,
            schema_overrides={"phecode": str},
            try_parse_dates=True
        )

        # Load covariate data
        # Make sure person_id in covariate data has the same type as person_id in phecode count
        cohort_sep = _utils.detect_delimiter(cohort_file_path)
        self.covariate_df = pl.read_csv(
            cohort_file_path,
            separator=cohort_sep,
            try_parse_dates=True
        )

        # Basic attributes from instantiation
        self.sex_at_birth_col = sex_at_birth_col
        if isinstance(covariate_cols, str):
            self.covariate_cols = [copy.deepcopy(covariate_cols)]
        elif isinstance(covariate_cols, list):
            self.covariate_cols = copy.deepcopy(covariate_cols)
        self.independent_variable_of_interest = independent_variable_of_interest
        self.verbose = verbose
        self.min_cases = min_cases
        self.min_phecode_count = min_phecode_count
        self.suppress_warnings = suppress_warnings
        self.method = method
        # Set default batch_size based on method if None
        if batch_size is None:
            if method == "logit":
                self.batch_size = 1
            elif method == "cox":
                self.batch_size = 10
            else:
                self.batch_size = 1  # fallback default
        else:
            self.batch_size = batch_size
        self.fall_back_to_serial = fall_back_to_serial

        # For Cox regression:
        if (method == "cox") and (
                (cox_control_observed_time_col is None) |
                (cox_phecode_observed_time_col is None)
        ):
            print()
            print("Warning: cox_control_observed_time_col and cox_phecode_observed_time_col are all required for Cox regression.")
            print("Please provide these parameters when using --method cox")
            print()
            sys.exit(1)
        else:
            self.cox_control_observed_time_col = cox_control_observed_time_col
            self.cox_phecode_observed_time_col = cox_phecode_observed_time_col
        self.cox_start_date_col = cox_start_date_col
        self.cox_stratification_col = cox_stratification_col
        if cox_control_observed_time_col == sex_at_birth_col:
            self.cox_stratification_by_sex = True
        else:
            self.cox_stratification_by_sex = False
        self.cox_fallback_step_size = cox_fallback_step_size

        # Assign 1 & 0 to male and female based on the male_as_one parameter
        if male_as_one:
            self.male_value = 1
            self.female_value = 0
        else:
            self.male_value = 0
            self.female_value = 1

        # Exclusion:
        # - phecode 1.2: user can choose to use exclusion or not
        # - phecode X: exclusion is removed, therefore, this parameter will be False for Phecode X regardless of input
        # to prevent user error
        if phecode_version == "1.2":
            self.use_exclusion = use_exclusion
        elif phecode_version == "X":
            self.use_exclusion = False

        # Check if variable_of_interest is included in covariate_cols
        self.variable_of_interest_in_covariates = False
        if self.independent_variable_of_interest in self.covariate_cols:
            self.variable_of_interest_in_covariates = True
            self.covariate_cols.remove(self.independent_variable_of_interest)
            print()
            print(f"Note: \"{self.independent_variable_of_interest}\" will not be used as covariate",
                  "since it was already specified as variable of interest.")
            print()

        # Remove the sex_at_birth column from covariates if it is included, just for processing purpose
        self.sex_as_covariate = False
        if self.sex_at_birth_col in self.covariate_cols:
            self.sex_as_covariate = True
            self.covariate_cols.remove(self.sex_at_birth_col)

        # Check for sex in data
        self.data_has_single_sex = False
        self.gender_specific_var_cols = [self.independent_variable_of_interest] + self.covariate_cols
        self.sex_values = self.covariate_df[sex_at_birth_col].unique().to_list()
        # When cohort only has 1 sex with 0 or 1 as value
        if (len(self.sex_values) == 1) and ((0 in self.sex_values) or (1 in self.sex_values)):
            self.data_has_single_sex = True
            self.single_sex_value = self.covariate_df[sex_at_birth_col].unique().to_list()[0]
            self.var_cols = [self.independent_variable_of_interest] + self.covariate_cols
            # When cohort only has 1 sex, and sex was chosen as a covariate
            if self.sex_as_covariate:
                print()
                print(f"Note: \"{self.sex_at_birth_col}\" will not be used as covariate",
                      "since there is only one sex in data.")
                print()
            # When cohort only has 1 sex, and variable_of_interest is also sex_at_birth column
            if self.independent_variable_of_interest == self.sex_at_birth_col:
                print()
                print(f"Warning: Cannot use \"{self.sex_at_birth_col}\" as variable of interest in single sex cohorts.")
                print()
                sys.exit(1)
        # When cohort has 2 sexes
        elif len(self.sex_values) == 2 and ((0 in self.sex_values) and (1 in self.sex_values)):
            if self.independent_variable_of_interest == self.sex_at_birth_col:
                self.var_cols = self.covariate_cols + [self.sex_at_birth_col]
            else:
                if not self.sex_as_covariate:
                    print()
                    print("Warning: Data has both sexes but user did not specify sex as a covariate.")
                    print("         Running PheWAS without sex as a covariate.")
                    print()
                self.var_cols = [self.independent_variable_of_interest] + self.covariate_cols + [self.sex_at_birth_col]
        # All other cases where sex_at_birth_column not coded probably
        else:
            print(f"Warning: Please check column {self.sex_at_birth_col}.")
            print("          This column should have upto 2 unique values, 0 for female and 1 for male.")
            sys.exit(1)

        # Check for string type variables among covariates
        if pl.Utf8 in self.covariate_df[self.var_cols].schema.values():
            str_cols = [k for k, v in self.covariate_df.schema.items() if v is pl.Utf8]
            print(f"Column(s) {str_cols} contain(s) string type. Only numerical types are accepted.")
            sys.exit(1)

        # Keep only relevant columns in covariate_df
        cols_to_keep = list(set(["person_id"] + self.var_cols))
        # For Cox regression add stratification and observed time to covariate df
        if method == "cox":
            cols_to_keep = cols_to_keep + [cox_control_observed_time_col]
            if cox_start_date_col is not None:
                cols_to_keep = cols_to_keep + [cox_start_date_col]
            if (cox_stratification_col is not None) and (cox_stratification_col not in cols_to_keep):
                cols_to_keep = cols_to_keep + [cox_stratification_col]
        self.covariate_df = self.covariate_df[cols_to_keep]
        self.covariate_df = self.covariate_df.drop_nulls()
        self.cohort_size = self.covariate_df.n_unique()

        # Update phecode_counts to only participants of interest
        self.cohort_ids = self.covariate_df["person_id"].unique().to_list()
        self.phecode_counts = self.phecode_counts.filter(pl.col("person_id").is_in(self.cohort_ids))
        if (cox_start_date_col is not None) and (method == "cox"):
            self.phecode_counts = self.phecode_counts.join(
                self.covariate_df[["person_id", cox_start_date_col]], how="left", on="person_id"
            )
        if phecode_to_process is None:
            self.phecode_list = self.phecode_counts["phecode"].unique().to_list()
        else:
            if isinstance(phecode_to_process, str):
                phecode_to_process = [phecode_to_process]
            self.phecode_list = phecode_to_process
        self.phecode_batch_list = self._split_phecode_list(phecode_list=self.phecode_list, batch_size=self.batch_size)

        # Attributes for reporting PheWAS results
        self._phecode_summary_statistics = None
        self._cases = None
        self._controls = None
        self.results = None
        self.not_tested_count = 0
        self.tested_count = 0
        self.bonferroni = None
        self.phecodes_above_bonferroni = None
        self.above_bonferroni_count = None

        # For saving results
        if output_file_path is not None:
            if ".tsv" in output_file_path:
                output_file_path = output_file_path.replace(".tsv", "")
            self.output_file_path = output_file_path + ".tsv"
        else:
            self._timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_file_path = f"phewas_{self._timestamp}.tsv"

        # Print some description
        print("Cohort size: ", self.cohort_size)
        if self.covariate_df[self.independent_variable_of_interest].n_unique() < 10:
            print(
                f"{self.independent_variable_of_interest} descriptions: ",
                self.covariate_df.group_by(self.independent_variable_of_interest).len(name="count")
            )
        else:
            print(
                f"{self.independent_variable_of_interest} descriptions: ",
                self.covariate_df[self.independent_variable_of_interest].describe()
            )
        print()
        print("Number of unique phecodes in cohort: ", len(self.phecode_list))
        print("Total number of phecode events: ", self.phecode_counts.shape[0])
        print("Number of phecode batches to process: ", len(self.phecode_batch_list))
        print()
        method_text = ""
        if self.method == "cox":
            method_text = "Cox regression"
        elif self.method == "logit":
            method_text = "Logistic regression"
        print("Analysis method: ", method_text)
        print()
        
        # Create temporary directory for batch results
        self.temp_dir = "/tmp/phewas_results"
        os.makedirs(self.temp_dir, exist_ok=True)

    @staticmethod
    def _to_polars(df: pd.DataFrame | pl.DataFrame) -> pl.DataFrame:
        """
        Convert pandas DataFrame to polars DataFrame if necessary.
        
        :param df: Input dataframe to convert.
        :type df: pd.DataFrame | pl.DataFrame
        :return: Polars DataFrame object.
        :rtype: pl.DataFrame
        """
        if isinstance(df, pd.DataFrame):
            polars_df = pl.from_pandas(df)
        else:
            polars_df = df
        return polars_df

    @staticmethod
    def _split_phecode_list(
            phecode_list: list[str], 
            batch_size: int
    ) -> list[list[str]]:
        """
        Split list of phecodes into smaller batches for parallel processing.
        
        Divides the complete phecode list into sublists of specified batch size
        to enable efficient parallel execution across multiple workers.
        
        :param phecode_list: Complete list of phecodes to process.
        :type phecode_list: list[str]
        :param batch_size: Maximum number of phecodes per batch.
        :type batch_size: int
        :return: List containing batches of phecodes.
        :rtype: list[list[str]]
        """
        sublists = []
        for i in range(0, len(phecode_list), batch_size):
            # Limit the end index to avoid out-of-bounds access
            end_idx = min(i + batch_size, len(phecode_list))
            sublists.append(phecode_list[i:end_idx])
        return sublists

    def _exclude_range(
            self, 
            phecode: str, 
            phecode_df: pl.DataFrame | None = None
    ) -> list[str]:
        """
        Process exclude_range text data for a specific phecode.
        
        Parses the exclude_range column from phecode mapping to extract
        individual phecodes or phecode ranges that should be excluded when
        defining controls. Handles single codes, ranges (e.g., "777-780"),
        and comma-separated combinations.
        
        :param phecode: Target phecode to get exclusion range for.
        :type phecode: str
        :param phecode_df: Phecode mapping dataframe, uses instance default if None.
        :type phecode_df: pl.DataFrame | None
        :return: List of phecodes to exclude when defining controls.
        :rtype: list[str]
        """
        if phecode_df is None:
            phecode_df = self.phecode_df

        # Not all phecode has exclude_range
        # Exclude_range can be single code (e.g., "777"), single range (e.g., "777-780"),
        # or multiple ranges/codes (e.g., "750-777,586.2")
        phecodes_without_exclude_range = phecode_df.filter(
            pl.col("exclude_range").is_null()
        )["phecode"].unique().to_list()
        if phecode in phecodes_without_exclude_range:
            exclude_range = []
        else:
            # Get exclude_range value of phecode
            ex_val = phecode_df.filter(pl.col("phecode") == phecode)["exclude_range"].unique().to_list()[0]

            # Split multiple codes/ranges
            comma_split = ex_val.split(",")
            if len(comma_split) == 1:
                exclude_range = comma_split
            else:
                exclude_range = []
                for item in comma_split:
                    # Process range in text form
                    if "-" in item:
                        first_code = item.split("-")[0]
                        last_code = item.split("-")[1]
                        dash_range = [str(i) for i in range(int(first_code), int(last_code))] + [last_code]
                        exclude_range = exclude_range + dash_range
                    else:
                        exclude_range = exclude_range + [item]

        return exclude_range

    def _case_control_prep(
            self, 
            phecode: str,
            phecode_counts: pl.DataFrame | None = None, 
            covariate_df: pl.DataFrame | None = None, 
            phecode_df: pl.DataFrame | None = None,
            var_cols: list[str] | None = None, 
            gender_specific_var_cols: list[str] | None = None, 
            keep_ids: bool = False
    ) -> tuple[pl.DataFrame, pl.DataFrame, list[str]]:
        """
        Prepare case and control cohorts for a specific phecode analysis.
        
        Filters cohort data based on phecode sex restrictions, identifies cases
        (participants with sufficient phecode counts) and controls (participants
        without the phecode or excluded conditions), handles Cox regression
        time variables, and prepares final datasets for statistical analysis.
        
        :param phecode: Target phecode for case-control preparation.
        :type phecode: str
        :param phecode_counts: Phecode count data, uses instance default if None.
        :type phecode_counts: pl.DataFrame | None
        :param covariate_df: Cohort covariate data, uses instance default if None.
        :type covariate_df: pl.DataFrame | None
        :param phecode_df: Phecode mapping data, uses instance default if None.
        :type phecode_df: pl.DataFrame | None
        :param var_cols: Variable columns for mixed-sex analysis, uses instance default if None.
        :type var_cols: list[str] | None
        :param gender_specific_var_cols: Variable columns for sex-specific analysis, uses instance default if None.
        :type gender_specific_var_cols: list[str] | None
        :param keep_ids: If True, retain person_id column in output datasets.
        :type keep_ids: bool
        :return: Cases dataframe, controls dataframe, and analysis variable columns.
        :rtype: tuple[pl.DataFrame, pl.DataFrame, list[str]]
        """
        if phecode_counts is None:
            phecode_counts = self.phecode_counts
        if covariate_df is None:
            covariate_df = self.covariate_df
        if phecode_df is None:
            phecode_df = self.phecode_df
        if var_cols is None:
            var_cols = copy.deepcopy(self.var_cols)
        if gender_specific_var_cols is None:
            gender_specific_var_cols = copy.deepcopy(self.gender_specific_var_cols)

        sex_as_covariate = copy.deepcopy(self.sex_as_covariate)
        data_has_single_sex = copy.deepcopy(self.data_has_single_sex)
        sex_at_birth_col = copy.deepcopy(self.sex_at_birth_col)
        male_value = copy.deepcopy(self.male_value)
        female_value = copy.deepcopy(self.female_value)
        sex_values = copy.deepcopy(self.sex_values)
        min_phecode_count = copy.deepcopy(self.min_phecode_count)
        use_exclusion = copy.deepcopy(self.use_exclusion)
        method = copy.deepcopy(self.method)
        cox_start_date_col = copy.deepcopy(self.cox_start_date_col)
        cox_phecode_observed_time_col = copy.deepcopy(self.cox_phecode_observed_time_col)
        cox_control_observed_time_col = copy.deepcopy(self.cox_control_observed_time_col)
        cox_stratification_col = copy.deepcopy(self.cox_stratification_col)

        # SEX RESTRICTION
        filtered_phecode_df = phecode_df.filter(pl.col("phecode") == phecode)
        if len(filtered_phecode_df["sex"].unique().to_list()) > 0:
            sex_restriction = filtered_phecode_df["sex"].unique().to_list()[0]
        else:
            return pl.DataFrame(), pl.DataFrame(), []

        # ANALYSIS VAR COLS
        if sex_restriction == "Both" and sex_as_covariate:
            analysis_var_cols = var_cols
        else:
            analysis_var_cols = gender_specific_var_cols

        # FILTER COVARIATE DATA BY PHECODE SEX RESTRICTION
        if not data_has_single_sex:
            # Filter cohort for just sex of phecode
            if sex_restriction == "Male":
                covariate_df = covariate_df.filter(pl.col(sex_at_birth_col) == male_value)
            elif sex_restriction == "Female":
                covariate_df = covariate_df.filter(pl.col(sex_at_birth_col) == female_value)
        else:
            # If data has single sex and different from sex of phecode, return empty dfs
            # otherwise, data is fine as is, i.e., nothing needed to be done
            if (
                (sex_restriction == "Male" and male_value not in sex_values)
                or (sex_restriction == "Male") and (male_value not in sex_values)
            ):
                return pl.DataFrame(), pl.DataFrame(), []

        # Exclude participants with existing condition if a cox_start_date_col is provided
        # (cox_start_date_col is not None and method = "cox" during class PheWAS instantiation)
        if cox_start_date_col in phecode_counts.columns:
            cox_exclude_ids = phecode_counts.filter(
                (pl.col("phecode") == phecode) &
                (pl.col("first_event_date") < pl.col(cox_start_date_col))
            )["person_id"].unique().to_list()
        else:
            cox_exclude_ids = None
        if cox_exclude_ids is not None:
            phecode_counts = phecode_counts.filter(
                ~pl.col("person_id").is_in(cox_exclude_ids)
            )

        # GENERATE CASES & CONTROLS
        if len(covariate_df) > 0:
            # CASES
            # Participants with at least <min_phecode_count> phecodes
            case_ids = phecode_counts.filter(
                (pl.col("phecode") == phecode) & (pl.col("count") >= min_phecode_count)
            )["person_id"].unique().to_list()
            cases = covariate_df.filter(pl.col("person_id").is_in(case_ids))

            # CONTROLS
            # Phecode exclusions
            if use_exclusion:
                exclude_range = [phecode] + self._exclude_range(phecode, phecode_df=phecode_df)
            else:
                exclude_range = [phecode]
            # Get participants ids to exclude and filter covariate df
            exclude_ids = phecode_counts.filter(
                pl.col("phecode").is_in(exclude_range)
            )["person_id"].unique().to_list()
            controls = covariate_df.filter(~(pl.col("person_id").is_in(exclude_ids)))

            # PROCESS OBSERVED TIME FOR COX REGRESSION
            if method == "cox":
                # CASES
                case_observed_time_df = phecode_counts.filter(
                    (pl.col("person_id").is_in(case_ids)) & (pl.col("phecode") == phecode)
                )[["person_id", cox_phecode_observed_time_col]]
                cases = cases.join(
                    case_observed_time_df, how="left", on="person_id"
                ).drop(
                    cox_control_observed_time_col
                ).rename(
                    {cox_phecode_observed_time_col: "observed_time"}
                )

                # CONTROLS
                controls = controls.rename({cox_control_observed_time_col: "observed_time"})

            # KEEP ONLY REQUIRED COLUMNS
            if method == "cox":
                analysis_var_cols = analysis_var_cols + ["observed_time"]
                if ((sex_restriction == "Both") and
                    ((cox_stratification_col is not None) and
                     (cox_stratification_col not in analysis_var_cols))):
                    analysis_var_cols = analysis_var_cols + [cox_stratification_col]

            # DUPLICATE CHECK
            # Drop duplicates
            duplicate_check_cols = ["person_id"] + analysis_var_cols
            cases = cases.unique(subset=duplicate_check_cols)
            controls = controls.unique(subset=duplicate_check_cols)

            # This only affects get_phecode_data since keep_ids is always False everywhere else
            if not keep_ids:
                # KEEP ONLY REQUIRED COLUMNS
                cases = cases[analysis_var_cols]
                controls = controls[analysis_var_cols]
            else:
                cases = cases[duplicate_check_cols]
                controls = controls[duplicate_check_cols]

            return cases, controls, analysis_var_cols

        else:
            return pl.DataFrame(), pl.DataFrame(), []

    def get_phecode_data(
            self, 
            phecode: str
    ) -> pl.DataFrame | None:
        """
        Retrieve combined case-control dataset for a specific phecode.
        
        Creates a unified dataset containing both cases and controls with
        an indicator column (is_phecode_case) to distinguish between them.
        Useful for external analysis or data exploration.
        
        :param phecode: Target phecode to retrieve data for.
        :type phecode: str
        :return: Combined dataset with cases and controls, or None if no data available.
        :rtype: pl.DataFrame | None
        """
        cases, controls, analysis_var_cols = self._case_control_prep(
            phecode=phecode,
            keep_ids=True
        )

        if len(cases) >= 0 or len(controls) >= 0:
            cases = cases.with_columns(pl.lit(True).alias("is_phecode_case"))
            controls = controls.with_columns(pl.lit(False).alias("is_phecode_case"))
            phecode_data = cases.vstack(controls)
            return phecode_data

        else:
            print(f"No phecode data for {phecode}")
            return None

    @staticmethod
    def _logit_result_prep(
            result, 
            var_of_interest_index: int
    ) -> dict[str, float | str]:
        """
        Extract and format key statistics from logistic regression results.
        
        Processes statsmodels logistic regression output to extract p-values,
        confidence intervals, odds ratios, and convergence information for
        the variable of interest.
        
        :param result: Statsmodels logistic regression result object.
        :param var_of_interest_index: Index position of variable of interest in model.
        :type var_of_interest_index: int
        :return: Dictionary containing formatted statistical results.
        :rtype: dict[str, float | str]
        """
        results_as_html = result.summary().tables[0].as_html()
        converged = pd.read_html(StringIO(results_as_html))[0].iloc[5, 1]
        results_as_html = result.summary().tables[1].as_html()
        res = pd.read_html(StringIO(results_as_html), header=0, index_col=0)[0]

        p_value = result.pvalues[var_of_interest_index]
        neg_log_p_value = -np.log10(p_value)
        standard_error = res.iloc[var_of_interest_index]['std err']
        beta = result.params[var_of_interest_index]
        conf_int_1 = res.iloc[var_of_interest_index]['[0.025']
        conf_int_2 = res.iloc[var_of_interest_index]['0.975]']
        odds_ratio = np.exp(beta)
        log10_odds_ratio = np.log10(odds_ratio)

        return {
            "p_value": p_value,
            "neg_log_p_value": neg_log_p_value,
            "standard_error": standard_error,
            "beta": beta,
            "conf_int_1": conf_int_1,
            "conf_int_2": conf_int_2,
            "odds_ratio": odds_ratio,
            "log10_odds_ratio": log10_odds_ratio,
            "converged": converged
        }

    def _cox_result_prep(
            self, 
            result, 
            stratified_by: str, 
            warning_message: str | None = None
    ) -> dict[str, float | str]:
        """
        Extract and format key statistics from Cox regression results.
        
        Processes lifelines Cox proportional hazards model output to extract
        hazard ratios, confidence intervals, p-values, concordance index,
        and convergence information for the variable of interest.
        
        :param result: Lifelines CoxPHFitter result object.
        :param stratified_by: Name of stratification variable or "None" if unstratified.
        :type stratified_by: str
        :param warning_message: Convergence or warning messages to include in results.
        :type warning_message: str | None
        :return: Dictionary containing formatted Cox regression statistics.
        :rtype: dict[str, float | str]
        """
        result_df = result.summary

        p_value = result_df.loc[self.independent_variable_of_interest]["p"]
        neg_log_p_value = -np.log10(p_value)
        standard_error = result_df.loc[self.independent_variable_of_interest]["se(coef)"]
        hazard_ratio = result_df.loc[self.independent_variable_of_interest]["exp(coef)"]
        hazard_ratio_low = result_df.loc[self.independent_variable_of_interest]["exp(coef) lower 95%"]
        hazard_ratio_high = result_df.loc[self.independent_variable_of_interest]["exp(coef) upper 95%"]
        log_hazard_ratio = result_df.loc[self.independent_variable_of_interest]["coef"]

        concordance_index = result.concordance_index_
        stratified_by = stratified_by

        result_dict = {
            "p_value": p_value,
            "neg_log_p_value": neg_log_p_value,
            "standard_error": standard_error,
            "hazard_ratio": hazard_ratio,
            "hazard_ratio_low": hazard_ratio_low,
            "hazard_ratio_high": hazard_ratio_high,
            "log_hazard_ratio": log_hazard_ratio,
            "concordance_index": concordance_index,
            "stratified_by": stratified_by,
            "convergence": warning_message
        }

        return result_dict

    # noinspection PyInconsistentReturns
    def _regression(
            self, 
            phecode: str
    ) -> dict[str, float | str | int] | None:
        """
        Perform regression analysis (logistic or Cox) for a single phecode.
        
        Conducts the complete statistical analysis pipeline for one phecode:
        prepares case-control data, fits the specified regression model,
        handles convergence issues, and formats results. Returns None if
        insufficient cases/controls or if model fitting fails.
        
        :param phecode: Target phecode for regression analysis.
        :type phecode: str
        :return: Formatted regression results or None if analysis failed.
        :rtype: dict[str, float | str | int] | None
        """

        phecode_counts = self.phecode_counts
        covariate_df = self.covariate_df
        var_cols = copy.deepcopy(self.var_cols)
        gender_specific_var_cols = copy.deepcopy(self.gender_specific_var_cols)
        min_cases = copy.deepcopy(self.min_cases)
        suppress_warnings = copy.deepcopy(self.suppress_warnings)
        method = copy.deepcopy(self.method)
        verbose = copy.deepcopy(self.verbose)
        independent_variable_of_interest = copy.deepcopy(self.independent_variable_of_interest)
        cox_stratification_col = copy.deepcopy(self.cox_stratification_col)
        cox_fallback_step_size = copy.deepcopy(self.cox_fallback_step_size)

        cases, controls, analysis_var_cols = self._case_control_prep(phecode,
                                                                     phecode_counts=phecode_counts,
                                                                     covariate_df=covariate_df,
                                                                     var_cols=var_cols,
                                                                     gender_specific_var_cols=gender_specific_var_cols)

        # Only run regression if number of cases > min_cases
        if (len(cases) >= min_cases) and (len(controls) > min_cases):

            if suppress_warnings:
                warnings.simplefilter("ignore")
                # Specifically suppress pandas BlockManager warnings
                warnings.filterwarnings("ignore", category=FutureWarning, message=".*BlockManager.*")
            else:
                warnings.simplefilter("always")

            # Add case/control values
            cases = cases.with_columns(pl.Series([1] * len(cases)).alias("y"))
            controls = controls.with_columns(pl.Series([0] * len(controls)).alias("y"))

            # Merge cases & controls
            regressors = cases.vstack(controls)

            # Convert to numpy series
            y = regressors["y"].to_numpy()

            # Base result dict
            base_dict = {"phecode": phecode,
                         "cases": len(cases),
                         "controls": len(controls)}

            # OPTION 1: COX REGRESSION
            if method == "cox":
                # For cox regression, warnings are always on to catch convergence status
                warnings.simplefilter("always")

                strata = None
                stratified_by = "None"
                if cox_stratification_col in regressors.columns:
                    strata = stratified_by = cox_stratification_col
                cox = CoxPHFitter()
                combined_warning = "Converged"
                try:
                    # Wrap fit() in warning handler
                    captured_warnings = []
                    with warnings.catch_warnings(record=True) as w:
                        result = cox.fit(
                            df=regressors.to_pandas(use_pyarrow_extension_array=True),
                            event_col="y",
                            duration_col="observed_time",
                            strata=strata,
                        )
                    for warning in w:
                        warning_message = str(warning.message)
                        captured_warnings.append(warning_message)
                    if captured_warnings:
                        combined_warning = "\n".join(captured_warnings)
                except u.ConvergenceError:
                    combined_warning = f"Convergence error. step_size was lowered to {cox_fallback_step_size} (default is 0.95)."
                    result = cox.fit(
                        df=regressors.to_pandas(use_pyarrow_extension_array=True),
                        event_col="y",
                        duration_col="observed_time",
                        strata=strata,
                        fit_options={"step_size": cox_fallback_step_size}
                    )
                except Exception as e:
                    print("Exception:", e)
                    result = None

                # Process result
                if result is not None:
                    stats_dict = self._cox_result_prep(
                        result,
                        stratified_by=stratified_by,
                        warning_message=combined_warning
                    )
                    result_dict = {**base_dict, **stats_dict}

                    # Choose to see results on the fly
                    if verbose:
                        print(f"Phecode {phecode} ({len(cases)} cases/{len(controls)} controls): {result_dict}\n")

                    return result_dict

            # OPTION 2: LOGISTIC REGRESSION
            if method == "logit":
                regressors = regressors[analysis_var_cols]
                # Get index of variable of interest
                var_index = regressors.columns.index(independent_variable_of_interest)
                regressors = regressors.to_numpy()
                regressors = sm.tools.add_constant(regressors, prepend=False)
                logit = sm.Logit(y, regressors, missing="drop")

                # Catch Singular matrix error
                try:
                    result = logit.fit(disp=False)
                except (np.linalg.linalg.LinAlgError, statsmodels.tools.sm_exceptions.PerfectSeparationError) as err:
                    if "Singular matrix" in str(err) or "Perfect separation" in str(err):
                        if verbose:
                            print(f"Phecode {phecode} ({len(cases)} cases/{len(controls)} controls):", str(err), "\n")
                        pass
                    else:
                        raise
                    result = None

                if result is not None:
                    # Process result
                    stats_dict = self._logit_result_prep(result=result, var_of_interest_index=var_index)
                    result_dict = {**base_dict, **stats_dict}  # python 3.5 or later
                    # result_dict = base_dict | stats_dict  # python 3.9 or later

                    # Choose to see results on the fly
                    if verbose:
                        print(f"Phecode {phecode} ({len(cases)} cases/{len(controls)} controls): {result_dict}\n")

                    return result_dict

        else:
            if verbose:
                print(f"Phecode {phecode} ({len(cases)} cases/{len(controls)} controls):",
                      "Not enough cases or controls. Pass.\n")

    def _combine_parquet_results(self, result_file_paths: list[str]) -> list[dict]:
        """
        Combine multiple parquet result files into a single list of dictionaries.
        
        Reads all parquet files, combines them using polars, cleans up temporary files,
        and returns the combined results as a list of dictionaries for compatibility
        with the rest of the analysis pipeline.
        
        :param result_file_paths: List of file paths to parquet result files.
        :type result_file_paths: list[str]
        :return: Combined results as list of dictionaries.
        :rtype: list[dict]
        """
        if not result_file_paths:
            return []
            
        print(f"Combining {len(result_file_paths)} result files...")
        result_dfs = []
        
        # Read all parquet files with progress tracking
        for file_path in tqdm(result_file_paths, desc="Reading files"):
            batch_df = pl.read_parquet(file_path)
            result_dfs.append(batch_df)
        
        # Combine all dataframes
        print("Concatenating results...")
        result_df = pl.concat(result_dfs)
        
        # Clean up temporary files
        print("Cleaning up temporary files...")
        for file_path in tqdm(result_file_paths, desc="Cleaning files"):
            try:
                os.remove(file_path)
            except OSError as e:
                if self.verbose:
                    print(f"Warning: Could not remove temp file {file_path}: {e}")
        
        # Try to remove temp directory if empty
        try:
            os.rmdir(self.temp_dir)
        except OSError:
            # Directory not empty or other issue - not critical
            if self.verbose:
                print(f"Note: Temp directory {self.temp_dir} not removed (may contain other files)")

        print()

        # Convert to list of dictionaries for compatibility
        return result_df.to_dicts()

    def _batch_regression(
            self, 
            phecode_batch: list[str]
    ) -> str:
        """
        Execute regression analysis for a batch of phecodes and save results to parquet.
        
        Processes multiple phecodes sequentially within a single worker,
        collecting successful results and saving them to a parquet file.
        Returns the path to the saved file for later combination.
        
        :param phecode_batch: List of phecodes to analyze in this batch.
        :type phecode_batch: list[str]
        :return: Path to the saved parquet file containing batch results.
        :rtype: str
        """
        results = []
        for phecode in phecode_batch:
            result = self._regression(phecode=phecode)
            if result is not None:
                results.append(result)
        
        # Save results to parquet file
        if results:
            # Generate timestamp-based filename for parallel safety
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            batch_filename = f"batch_{timestamp}.parquet"
            batch_filepath = os.path.join(self.temp_dir, batch_filename)
            
            # Convert to polars DataFrame and save
            batch_df = pl.from_dicts(results)
            batch_df.write_parquet(batch_filepath)
            
            return batch_filepath
        else:
            # Return empty string if no results to indicate empty batch
            return ""

    def generate_phewas_script(
            self,
            script_name: str = "phewas_script.sh",
            parallelization: str | None = None,
            n_workers: int | None = None
    ) -> None:
        """
        Generate a bash script for distributed PheWAS execution.
        
        Creates a shell script containing the exact command-line arguments
        needed to reproduce this PheWAS analysis configuration. Useful for
        distributed computing environments like Google Cloud Batch or dsub.
        
        :param script_name: Name of output bash script file.
        :type script_name: str
        :param parallelization: Parallelization method to include in script.
        :type parallelization: str | None
        :param n_workers: Number of workers to include in script.
        :type n_workers: int | None
        :return: Writes script file and prints content.
        :rtype: None
        """

        phewas_script = "phetk phewas"

        param_dict = {
            "--phecode_version": self.phecode_version,
            "--phecode_count_file_path": "$PHECODE_COUNT_FILE_PATH",  #  Must be a gcp bucket path
            "--cohort_file_path": "$COHORT_FILE_PATH",  #  Must be a gcp bucket path
            "--covariate_cols": " ".join(self.input_covariate_cols),
            "--independent_variable_of_interest": self.independent_variable_of_interest,
            "--sex_at_birth_col": self.sex_at_birth_col,
            "--male_as_one": self.male_as_one,
            "--output_file_path": "$OUTPUT_FILE_PATH",  #  Must be a gcp bucket path
            "--min_cases": self.min_cases,
            "--min_phecode_count": self.min_phecode_count,
            "--phecode_to_process": self.phecode_to_process,
            "--method": self.method,
            "--batch_size": self.batch_size,
        }
        # Add cox params when method = cox
        if self.method == "cox":
            if self.cox_start_date_col is not None:
                param_dict["--cox_start_date_col"] = self.cox_start_date_col
            if self.cox_control_observed_time_col is not None:
                param_dict["--cox_control_observed_time_col"] = self.cox_control_observed_time_col
            if self.cox_phecode_observed_time_col is not None:
                param_dict["--cox_phecode_observed_time_col"] = self.cox_phecode_observed_time_col
            if self.cox_stratification_col is not None:
                param_dict["--cox_stratification_col"] = self.cox_stratification_col
            if self.cox_fallback_step_size is not None:
                param_dict["--cox_fallback_step_size"] = self.cox_fallback_step_size
        # These params are optional; default values will be used without being listed
        # and will only be listed in the script if different from default values
        if self.icd_version != "US":
            param_dict["--icd_version"] = self.icd_version
        if self.phecode_map_file_path is not None:
            param_dict["--phecode_map_file_path"] = "$PHECODE_MAP_FILE_PATH"
        if self.use_exclusion:
            param_dict["--use_exclusion"] = "True"
        if self.verbose:
            param_dict["--verbose"] = "True"
        if not self.suppress_warnings:
            param_dict["--suppress_warnings"] = "False"
        # Finally, add parallelization and n_workers
        if n_workers is not None:
            param_dict["--n_workers"] = n_workers
        if parallelization is not None:
            param_dict["--parallelization"] = parallelization

        for k,v in param_dict.items():
            if v is not None:
                phewas_script += f" {k} {v}"

        _utils.generate_sh_script(script_name=script_name, commands=[phewas_script])

        print()
        print("PheWAS script content:")
        with open(script_name, "r") as f:
            print(f.read())
        print()

    def run_dsub(
            self,
            docker_image: str,
            job_script_name: str = "phewas_script.sh",
            job_name: str | None = None,
            input_dict: dict[str, str] | None = None,
            output_dict: dict[str, str] | None = None,
            env_dict: dict[str, str] | None = None,
            machine_type: str = "c2d-highcpu-4",
            disk_type: str | None = None,
            boot_disk_size: int = 50,
            disk_size: int = 256,
            region: str = "us-central1",
            provider: str = "google-batch",
            preemptible: bool = False,
            use_private_address: bool = True,
            parallelization: str | None = None,
            n_workers: int | None = None,
            custom_args: str | None = None,
            show_dsub_command: bool = True,
            use_aou_docker_prefix: bool = True,
    ) -> None:
        """
        Execute PheWAS analysis using Google Cloud dsub for distributed computing.
        
        Configures and launches a dsub job to run PheWAS analysis on Google Cloud
        infrastructure. Automatically generates the execution script and handles
        input/output file paths in cloud storage.
        
        :param docker_image: Docker image containing PheWAS dependencies.
        :type docker_image: str
        :param job_script_name: Name of bash script to execute in dsub.
        :type job_script_name: str
        :param job_name: Custom name for dsub job.
        :type job_name: str | None
        :param input_dict: Mapping of input variable names to cloud storage paths.
        :type input_dict: dict[str, str] | None
        :param output_dict: Mapping of output variable names to cloud storage paths.
        :type output_dict: dict[str, str] | None
        :param env_dict: Environment variables to set in job.
        :type env_dict: dict[str, str] | None
        :param machine_type: Google Cloud machine type for job.
        :type machine_type: str
        :param disk_type: Type of persistent disk to attach.
        :type disk_type: str | None
        :param boot_disk_size: Size of boot disk in GB.
        :type boot_disk_size: int
        :param disk_size: Size of additional disk in GB.
        :type disk_size: int
        :param region: Google Cloud region for job execution.
        :type region: str
        :param provider: Cloud provider backend ("google-batch" or "google-v2").
        :type provider: str
        :param preemptible: Whether to use preemptible instances.
        :type preemptible: bool
        :param use_private_address: Whether to use private IP addresses only.
        :type use_private_address: bool
        :param parallelization: Parallelization method for analysis.
        :type parallelization: str | None
        :param n_workers: Number of workers for parallel processing.
        :type n_workers: int | None
        :param custom_args: Additional dsub command line arguments.
        :type custom_args: str | None
        :param show_dsub_command: Whether to display dsub command before execution.
        :type show_dsub_command: bool
        :param use_aou_docker_prefix: Whether to use All of Us docker registry prefix.
        :type use_aou_docker_prefix: bool
        :return: Launches dsub job and stores job object in self.dsub.
        :rtype: None
        """
        _utils.print_banner("Setting up dsub")
        print()

        # input_dict
        if input_dict is None:
            input_dict = {
                "PHECODE_COUNT_FILE_PATH": self.phecode_count_file_path,
                "COHORT_FILE_PATH": self.cohort_file_path,
            }
            if self.phecode_map_file_path is not None:
                input_dict["PHECODE_MAP_FILE_PATH"] = self.phecode_map_file_path

        # output_dict
        if output_dict is None:
            output_dict = {
                "OUTPUT_FILE_PATH": self.output_file_path,
            }

        # generate phewas script to run in dsub workers
        self.generate_phewas_script(
            script_name=job_script_name,
            parallelization=parallelization,
            n_workers=n_workers
        )

        # run dsub
        # noinspection PyUnresolvedReferences,PyProtectedMember
        from phetk import _dsub
        self.dsub = _dsub.Dsub(
            docker_image=docker_image,
            job_script_name=job_script_name,
            job_name=job_name,
            input_dict=input_dict,
            output_dict=output_dict,
            env_dict=env_dict,
            machine_type=machine_type,
            disk_type=disk_type,
            boot_disk_size=boot_disk_size,
            disk_size=disk_size,
            region=region,
            provider=provider,
            preemptible=preemptible,
            use_private_address=use_private_address,
            log_file_path = None,
            custom_args=custom_args,
            use_aou_docker_prefix=use_aou_docker_prefix,
        )
        self.dsub.run(show_command=show_dsub_command)
        
        # Save dsub instance as pickle for later access
        dsub_pickle_path = f"dsub_{self.dsub.job_name}.pkl"
        _utils.save_pickle_object(self.dsub, dsub_pickle_path)
        print()
        print(f"Dsub instance saved as '{dsub_pickle_path}'")
        print()
        print(f"To load and monitor this job later, run:")
        print("from phetk._utils import load_dsub_instance")
        print(f"dsub_instance = load_dsub_instance('{dsub_pickle_path}')")
        print()

    # noinspection PyUnreachableCode
    def run(
            self,
            parallelization: str | None = None,
            n_workers: int | None = None
    ) -> str | None:
        """
        Execute the complete PheWAS analysis across all phecodes.
        
        Performs regression analysis for all phecodes in the cohort using the
        specified parallelization method. Automatically selects optimal
        parallelization (multithreading for logistic, multiprocessing for Cox)
        if not specified. Saves results to file and reports summary statistics.
        
        :param parallelization: Execution method ("serial", "multithreading", "multiprocessing", or None for auto-selection).
        :type parallelization: str | None
        :param n_workers: Number of parallel workers, auto-determined if None.
        :type n_workers: int | None
        :return: Error message if invalid parallelization method, otherwise None.
        :rtype: str | None
        """

        # Assign an optimal parallelization method when it is not specified
        if parallelization is None:
            if self.method == "logit":
                parallelization = "multithreading"
            elif self.method == "cox":
                parallelization = "multiprocessing"

        _utils.print_banner("Running PheWAS")
        print()
        print("Parallelization method:", parallelization)

        result_dicts = []
        if parallelization == "serial":
            for phecode in tqdm(self.phecode_list, desc="Processed"):
                result = self._regression(
                    phecode=phecode
                )
                result_dicts.append(result)

        elif parallelization == "multithreading":
            if n_workers is None:
                n_workers = round(os.cpu_count()*2/3)
            print("Number of workers:", n_workers)
            print()
            print("Creating ThreadPoolExecutor...")
            result_file_paths = []
            try:
                with ThreadPoolExecutor(max_workers=n_workers) as executor:
                    if self.verbose:
                        print(f"ThreadPoolExecutor created with {n_workers} workers")
                    print("Submitting jobs to workers...")
                    jobs = [
                        executor.submit(
                            self._batch_regression,
                            phecode_batch
                        ) for phecode_batch in self.phecode_batch_list
                    ]
                    print(f"Submitted {len(jobs)} jobs. Running regressions...")
                    if self.verbose:
                        print("Waiting for first job completion...")
                    completed_count = 0
                    for job in tqdm(as_completed(jobs), total=len(self.phecode_batch_list), desc="Processed"):
                        completed_count += 1
                        if self.verbose:
                            print(f"Job {completed_count}/{len(jobs)} completed")
                        file_path = job.result()
                        if file_path:  # Only add non-empty file paths
                            result_file_paths.append(file_path)
                        if self.verbose:
                            if file_path:
                                temp_df = pl.read_parquet(file_path)
                                print(f"Job {completed_count} saved {len(temp_df)} results to {os.path.basename(file_path)}")
                            else:
                                print(f"Job {completed_count} had no results to save")
                print("Multithreading completed successfully.")
                
                # Combine parquet files
                result_dicts = self._combine_parquet_results(result_file_paths)
            except Exception as e:
                print(f"Error in multithreading: {e}")
                if self.verbose:
                    import traceback
                    print("Full traceback:")
                    traceback.print_exc()
                if self.fall_back_to_serial:
                    print("Falling back to serial processing...")
                    for phecode in tqdm(self.phecode_list, desc="Processed"):
                        result = self._regression(phecode=phecode)
                        if result is not None:
                            result_dicts.append(result)
                else:
                    print("Fallback to serial processing is disabled. Exiting.")
                    raise e

        elif parallelization == "multiprocessing":
            if n_workers is None:
                n_workers = os.cpu_count() - 1
            print("Number of workers:", n_workers)
            print()
            print("Initializing multiprocessing context...")
            result_file_paths = []
            try:
                mp_context = get_context("spawn")
                if self.verbose:
                    print("Multiprocessing context created successfully")
                print("Creating ProcessPoolExecutor...")
                with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp_context) as executor:
                    if self.verbose:
                        print(f"ProcessPoolExecutor created with {n_workers} workers")
                    print("Submitting jobs to workers...")
                    jobs = [
                        executor.submit(
                            self._batch_regression,
                            phecode_batch
                        ) for phecode_batch in self.phecode_batch_list
                    ]
                    print(f"Submitted {len(jobs)} jobs. Running regressions...")
                    if self.verbose:
                        print("Waiting for first job completion...")
                    completed_count = 0
                    for job in tqdm(as_completed(jobs), total=len(self.phecode_batch_list), desc="Processed"):
                        completed_count += 1
                        if self.verbose:
                            print(f"Job {completed_count}/{len(jobs)} completed")
                        file_path = job.result()
                        if file_path:  # Only add non-empty file paths
                            result_file_paths.append(file_path)
                        if self.verbose:
                            if file_path:
                                temp_df = pl.read_parquet(file_path)
                                print(f"Job {completed_count} saved {len(temp_df)} results to {os.path.basename(file_path)}")
                            else:
                                print(f"Job {completed_count} had no results to save")
                print("Multiprocessing completed successfully.")
                
                # Combine parquet files
                result_dicts = self._combine_parquet_results(result_file_paths)
            except Exception as e:
                print(f"Error in multiprocessing: {e}")
                if self.verbose:
                    import traceback
                    print("Full traceback:")
                    traceback.print_exc()
                if self.fall_back_to_serial:
                    print("Falling back to serial processing...")
                    for phecode in tqdm(self.phecode_list, desc="Processed"):
                        result = self._regression(phecode=phecode)
                        if result is not None:
                            result_dicts.append(result)
                else:
                    print("Fallback to serial processing is disabled. Exiting.")
                    raise e

        else:
            return "Invalid parallelization method! Select \"multithreading\", \"multiprocessing\", or \"serial\"."

        result_dicts = [result for result in result_dicts if result is not None]

        if len(result_dicts) > 0:
            result_df = pl.from_dicts(result_dicts)
            self.results = result_df.join(self.phecode_df[["phecode", "sex",
                                                           "phecode_string",
                                                           "phecode_category"]].unique(),
                                          how="left",
                                          on="phecode").rename({"sex": "phecode_sex_restriction"})

            _utils.print_banner("PheWAS Completed")
            print()

            self.tested_count = len(self.results)
            self.not_tested_count = len(self.phecode_list) - self.tested_count
            self.bonferroni = -np.log10(0.05 / self.tested_count)
            self.phecodes_above_bonferroni = self.results.filter(pl.col("neg_log_p_value") > self.bonferroni)
            self.above_bonferroni_count = len(self.phecodes_above_bonferroni)

            # Save results
            self.results.write_csv(self.output_file_path, separator="\t")

            print("Number of participants in cohort:", self.cohort_size)
            print("Number of phecodes in cohort:", len(self.phecode_list))
            print(f"Number of phecodes having less than {self.min_cases} cases or controls:", self.not_tested_count)
            print("Number of phecodes tested:", self.tested_count)
            print(u"Suggested Bonferroni correction (-log\u2081\u2080 scale):", self.bonferroni)
            print("Number of phecodes above Bonferroni correction:", self.above_bonferroni_count)
            print()
            print("PheWAS results saved to\033[1m", self.output_file_path, "\033[0m")
        else:
            print("No analysis done. Please check your PheWAS settings/inputs.")

        print()
        return None


def main() -> None:
    """
    Command-line interface entry point for PheWAS analysis.
    
    Parses command-line arguments, initializes PheWAS object with specified
    parameters, and executes the analysis. Handles both logistic and Cox
    regression methods with full parameter validation.
    
    :return: Executes PheWAS analysis and saves results to file.
    :rtype: None
    """

    # Parse args
    parser = argparse.ArgumentParser(description="PheWAS analysis tool.")
    parser.add_argument("--phecode_count_file_path",
                        type=str, required=True,
                        help="Path to phecode count csv/tsv file.")
    parser.add_argument("--cohort_file_path",
                        type=str, required=True,
                        help="Path to cohort csv/tsv file.")
    parser.add_argument("--phecode_version",
                        type=str, required=True, choices=["1.2", "X"],
                        help="Phecode version.")
    parser.add_argument("--method",
                        type=str, required=False, default="logit", choices=["logit", "cox"],
                        help="Phecode regression method. Can be 'logit' or 'cox'.")
    parser.add_argument("--cox_start_date_col",
                        type=str, required=False,
                        help="Start date column for Cox regression.")
    parser.add_argument("--cox_control_observed_time_col",
                        type=str, required=False,
                        help="Observed time for controls in phecode regression. Right censored time.")
    parser.add_argument("--cox_phecode_observed_time_col",
                        type=str, required=False,
                        help="Observed time for cases in phecode regression. First phecode event time.")
    parser.add_argument("--cox_stratification_col",
                        type=str, required=False,
                        help="Stratification for cox regression.")
    parser.add_argument("--cox_fallback_step_size",
                        type=float, required=False, default=0.1,
                        help="Cox fallback step size used when regression fails to converge with the default step size of 0.95.")
    parser.add_argument("--covariate_cols",
                        nargs="+",
                        type=str, required=True,
                        help="List of covariate column names to use in PheWAS analysis.")
    parser.add_argument("--independent_variable_of_interest",
                        type=str, required=True,
                        help="Independent variable of interest.")
    parser.add_argument("--sex_at_birth_col",
                        type=str, required=True,
                        help="Sex at birth column.")
    parser.add_argument("--male_as_one",
                        type=_utils.str_to_bool, required=False, default=True,
                        help="Whether male was assigned as 1 in data.")
    parser.add_argument("--phecode_to_process",
                        nargs="+",
                        type=str, required=False, default=None,
                        help="List of specific phecodes to use in PheWAS analysis.")
    parser.add_argument("--use_exclusion",
                        type=_utils.str_to_bool, required=False, default=False,
                        help="Whether to use phecode exclusions. Only applicable for phecode 1.2.")
    parser.add_argument("--min_cases",
                        type=int, required=False, default=50,
                        help="Minimum number of cases required to be tested.")
    parser.add_argument("--min_phecode_count",
                        type=int, required=False, default=2,
                        help="Minimum number of phecode counts required to be considered as case.")
    parser.add_argument("--icd_version",
                        type=str, required=False, default="US",
                        help="ICD version ('US', 'WHO', or 'custom').")
    parser.add_argument("--phecode_map_file_path",
                        type=str, required=False, default=None,
                        help="Path to custom phecode mapping file, required if icd_version='custom'.")
    parser.add_argument("--fall_back_to_serial",
                        type=_utils.str_to_bool, required=False, default=False,
                        help="Whether to fall back to serial processing on parallel failure.")
    parser.add_argument("--n_workers",
                        type=int, required=False, default=None,
                        help="Number of workers for parallel processing.")
    parser.add_argument("--output_file_path",
                        type=str, required=False, default=None)
    parser.add_argument("--parallelization",
                        type=str, required=False, default=None)
    parser.add_argument("--batch_size",
                        type=int, required=False, default=None, help="Batch size for parallelization. If None, defaults to 1 for logit and 10 for cox.")
    parser.add_argument("--suppress_warnings",
                        type=_utils.str_to_bool, required=False, default=True, help="Whether to suppress warnings.")
    parser.add_argument("--verbose",
                        type=_utils.str_to_bool, required=False, default=False, help="Whether to print verbose progress information.")
    args = parser.parse_args()

    # Run PheWAS
    phewas = PheWAS(
        phecode_version=args.phecode_version,
        phecode_count_file_path=args.phecode_count_file_path,
        cohort_file_path=args.cohort_file_path,
        sex_at_birth_col=args.sex_at_birth_col,
        male_as_one=args.male_as_one,
        covariate_cols=args.covariate_cols,
        independent_variable_of_interest=args.independent_variable_of_interest,
        cox_start_date_col=args.cox_start_date_col,
        cox_control_observed_time_col=args.cox_control_observed_time_col,
        cox_phecode_observed_time_col=args.cox_phecode_observed_time_col,
        cox_stratification_col=args.cox_stratification_col,
        cox_fallback_step_size=args.cox_fallback_step_size,
        icd_version=args.icd_version,
        phecode_map_file_path=args.phecode_map_file_path,
        phecode_to_process=args.phecode_to_process,
        use_exclusion=args.use_exclusion,
        min_cases=args.min_cases,
        min_phecode_count=args.min_phecode_count,
        output_file_path=args.output_file_path,
        method=args.method,
        batch_size=args.batch_size,
        fall_back_to_serial=args.fall_back_to_serial,
        suppress_warnings=args.suppress_warnings,
        verbose=args.verbose
    )
    phewas.run(parallelization=args.parallelization, n_workers=args.n_workers)


if __name__ == "__main__":
    main()
