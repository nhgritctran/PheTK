from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime
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
from PheTK import _utils


class PheWAS:

    def __init__(self,
                 phecode_version,
                 phecode_count_csv_path,
                 cohort_csv_path,
                 covariate_cols,
                 independent_variable_of_interest,
                 sex_at_birth_col,
                 male_as_one=True,
                 cox_control_observed_time_col=None,
                 cox_phecode_observed_time_col=None,
                 cox_stratification_col=None,
                 icd_version="US",
                 phecode_map_file_path=None,
                 phecode_to_process="all",
                 min_cases=50,
                 min_phecode_count=2,
                 use_exclusion=False,
                 output_file_name=None,
                 verbose=False,
                 suppress_warnings=True,
                 method="logit",
                 batch_size=1):
        """
        :param phecode_version: accepts "1.2" or "X"
        :param phecode_count_csv_path: path to phecode count of relevant participants at minimum
        :param cohort_csv_path: path to cohort data with covariates of interest
        :param sex_at_birth_col: gender/sex column of interest, by default, male = 1, female = 0;
                                 can use male_as_one parameter to specify the value assigned to male
        :param covariate_cols: name of covariate columns; excluding independent var of interest
        :param independent_variable_of_interest: independent variable of interest column name
        :param male_as_one: defaults to True; if True, male=1 and female=0; if False, male=0 and female=1;
                            use this to match how males and females are coded in sex_at_birth column;
        :param cox_control_observed_time_col: name of column in cohort dataframe,
                                              containing censoring time for controls in Cox regression.
        :param cox_phecode_observed_time_col: name of column in phecode counts dataframe,
                                              containing time to event (phecode) for cases in Cox regression.
        :param cox_stratification_col: name of column that is used fpr stratification in Cox regression.
        :param icd_version: defaults to "US"; other option are "WHO" and "custom";
                            if "custom", user need to provide phecode_map_path
        :param phecode_map_file_path: path to custom phecode map table
        :param phecode_to_process: defaults to "all"; otherwise, a list of phecodes must be provided
        :param min_cases: defaults to 50; minimum number of cases for each phecode to be considered for PheWAS
        :param min_phecode_count: defaults to 2; minimum number of phecode count to qualify as case for PheWAS
        :param use_exclusion: defaults to True for phecode 1.2; always False for phecode X;
                              whether to use additional exclusion range in control for PheWAS
        :param output_file_name: if None, defaults to "phewas_{timestamp}.csv"
        :param verbose: defaults to False; if True, print brief result of each phecode run
        :param method: defaults to "logit"; supports:
            "logit": logistic regression
            "cox": cox regression
        :param batch_size: defaults to 1; number of phecodes to be processed in each thread/process
        :param suppress_warnings: defaults to True;
                                  if True, ignore common exception warnings such as ConvergenceWarnings, etc.
        """
        print("~~~~~~~~~~~~~~~~~~~~~~~~    Creating PheWAS Object    ~~~~~~~~~~~~~~~~~~~~~~~~~~")

        # load phecode mapping file by version or by custom path
        self.phecode_df = _utils.get_phecode_mapping_table(
            phecode_version=phecode_version,
            icd_version=icd_version,
            phecode_map_file_path=phecode_map_file_path,
            keep_all_columns=True
        )

        # load phecode counts data for all participants
        # noinspection PyTypeChecker
        self.phecode_counts = pl.read_csv(
            phecode_count_csv_path,
            dtypes={"phecode": str},
            try_parse_dates=True
        )

        # load covariate data
        # make sure person_id in covariate data has the same type as person_id in phecode count
        self.covariate_df = pl.read_csv(cohort_csv_path)

        # basic attributes from instantiation
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
        self.batch_size = batch_size

        # for Cox regression:
        if (method == "cox") and ((cox_control_observed_time_col is None) | (cox_phecode_observed_time_col is None)):
            print()
            print("Warning: Both cox_observed_time_col and cox_phecode_observed_time_col are required for Cox "
                  "regression.")
            print()
            sys.exit(0)
        else:
            self.cox_control_observed_time_col = cox_control_observed_time_col
            self.cox_phecode_observed_time_col = cox_phecode_observed_time_col
        self.cox_stratification_col = cox_stratification_col
        if cox_control_observed_time_col == sex_at_birth_col:
            self.cox_stratification_by_sex = True
        else:
            self.cox_stratification_by_sex = False

        # assign 1 & 0 to male & female based on male_as_one parameter
        if male_as_one:
            self.male_value = 1
            self.female_value = 0
        else:
            self.male_value = 0
            self.female_value = 1

        # exclusion:
        # - phecode 1.2: user can choose to use exclusion or not
        # - phecode X: exclusion is removed, therefore this parameter will be False for Phecode X regardless of input
        # to prevent user error
        if phecode_version == "1.2":
            self.use_exclusion = use_exclusion
        elif phecode_version == "X":
            self.use_exclusion = False

        # check if variable_of_interest is included in covariate_cols
        self.variable_of_interest_in_covariates = False
        if self.independent_variable_of_interest in self.covariate_cols:
            self.variable_of_interest_in_covariates = True
            self.covariate_cols.remove(self.independent_variable_of_interest)
            print()
            print(f"Note: \"{self.independent_variable_of_interest}\" will not be used as covariate",
                  "since it was already specified as variable of interest.")
            print()

        # remove sex_at_birth column from covariates if it is included, just for processing purpose
        self.sex_as_covariate = False
        if self.sex_at_birth_col in self.covariate_cols:
            self.sex_as_covariate = True
            self.covariate_cols.remove(self.sex_at_birth_col)

        # check for sex in data
        self.data_has_single_sex = False
        self.gender_specific_var_cols = [self.independent_variable_of_interest] + self.covariate_cols
        self.sex_values = self.covariate_df[sex_at_birth_col].unique().to_list()
        # when cohort only has 1 sex with 0 or 1 as value
        if (len(self.sex_values) == 1) and ((0 in self.sex_values) or (1 in self.sex_values)):
            self.data_has_single_sex = True
            self.single_sex_value = self.covariate_df[sex_at_birth_col].unique().to_list()[0]
            self.var_cols = [self.independent_variable_of_interest] + self.covariate_cols
            # when cohort only has 1 sex, and sex was chosen as a covariate
            if self.sex_as_covariate:
                print()
                print(f"Note: \"{self.sex_at_birth_col}\" will not be used as covariate",
                      "since there is only one sex in data.")
                print()
            # when cohort only has 1 sex, and variable_of_interest is also sex_at_birth column
            if self.independent_variable_of_interest == self.sex_at_birth_col:
                print()
                print(f"Warning: Cannot use \"{self.sex_at_birth_col}\" as variable of interest in single sex cohorts.")
                print()
                sys.exit(0)
        # when cohort has 2 sexes
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
        # all other cases where sex_at_birth_column not coded probably
        else:
            print(f"Warning: Please check column {self.sex_at_birth_col}.")
            print("          This column should have upto 2 unique values, 0 for female and 1 for male.")
            sys.exit(0)

        # check for string type variables among covariates
        if pl.Utf8 in self.covariate_df[self.var_cols].schema.values():
            str_cols = [k for k, v in self.covariate_df.schema.items() if v is pl.Utf8]
            print(f"Column(s) {str_cols} contain(s) string type. Only numerical types are accepted.")
            sys.exit(0)

        # keep only relevant columns in covariate_df
        cols_to_keep = list(set(["person_id"] + self.var_cols))
        # for Cox regression add stratification & observed time to covariate df
        if method == "cox":
            cols_to_keep = cols_to_keep + [cox_control_observed_time_col]
            if (cox_stratification_col is not None) and (cox_stratification_col not in cols_to_keep):
                cols_to_keep = cols_to_keep + [cox_stratification_col]
        self.covariate_df = self.covariate_df[cols_to_keep]
        self.covariate_df = self.covariate_df.drop_nulls()
        self.cohort_size = self.covariate_df.n_unique()

        # update phecode_counts to only participants of interest
        self.cohort_ids = self.covariate_df["person_id"].unique().to_list()
        self.phecode_counts = self.phecode_counts.filter(pl.col("person_id").is_in(self.cohort_ids))
        if phecode_to_process == "all":
            self.phecode_list = self.phecode_counts["phecode"].unique().to_list()
        else:
            if isinstance(phecode_to_process, str):
                phecode_to_process = [phecode_to_process]
            self.phecode_list = phecode_to_process
        self.phecode_batch_list = self._split_phecode_list(phecode_list=self.phecode_list, batch_size=self.batch_size)

        # attributes for reporting PheWAS results
        self._phecode_summary_statistics = None
        self._cases = None
        self._controls = None
        self.results = None
        self.not_tested_count = 0
        self.tested_count = 0
        self.bonferroni = None
        self.phecodes_above_bonferroni = None
        self.above_bonferroni_count = None

        # for saving results
        if output_file_name is not None:
            if ".csv" in output_file_name:
                output_file_name = output_file_name.replace(".csv", "")
            self.output_file_name = output_file_name + ".csv"
        else:
            self._timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_file_name = f"phewas_{self._timestamp}.csv"

    @staticmethod
    def _to_polars(df):
        """
        Check and convert pandas dataframe object to polars dataframe, if applicable
        :param df: dataframe object
        :return: polars dataframe
        """
        if isinstance(df, pd.DataFrame):
            polars_df = pl.from_pandas(df)
        else:
            polars_df = df
        return polars_df

    @staticmethod
    def _split_phecode_list(phecode_list, batch_size):
        """
        Split phecode_list into batches of phecodes based on batch size. Used for parallel processing.
        :param phecode_list: list of all phecodes in cohort
        :param batch_size: phecode batch size
        :return: list of batches of phecodes
        """
        sublists = []
        for i in range(0, len(phecode_list), batch_size):
            # Limit the end index to avoid out-of-bounds access
            end_idx = min(i + batch_size, len(phecode_list))
            sublists.append(phecode_list[i:end_idx])
        return sublists

    def _exclude_range(self, phecode, phecode_df=None):
        """
        Process text data in exclude_range column; exclusively for phecodeX
        :param phecode: phecode of interest
        :return: processed exclude_range, either None or a valid list of phecode(s)
        """
        if phecode_df is None:
            phecode_df = self.phecode_df

        # not all phecode has exclude_range
        # exclude_range can be single code (e.g., "777"), single range (e.g., "777-780"),
        # or multiple ranges/codes (e.g., "750-777,586.2")
        phecodes_without_exclude_range = phecode_df.filter(
            pl.col("exclude_range").is_null()
        )["phecode"].unique().to_list()
        if phecode in phecodes_without_exclude_range:
            exclude_range = []
        else:
            # get exclude_range value of phecode
            ex_val = phecode_df.filter(pl.col("phecode") == phecode)["exclude_range"].unique().to_list()[0]

            # split multiple codes/ranges
            comma_split = ex_val.split(",")
            if len(comma_split) == 1:
                exclude_range = comma_split
            else:
                exclude_range = []
                for item in comma_split:
                    # process range in text form
                    if "-" in item:
                        first_code = item.split("-")[0]
                        last_code = item.split("-")[1]
                        dash_range = [str(i) for i in range(int(first_code), int(last_code))] + [last_code]
                        exclude_range = exclude_range + dash_range
                    else:
                        exclude_range = exclude_range + [item]

        return exclude_range

    def _case_control_prep(self, phecode,
                           phecode_counts=None, covariate_df=None, phecode_df=None,
                           var_cols=None, gender_specific_var_cols=None, keep_ids=False):
        """
        :param phecode: phecode of interest
        :param phecode_counts: phecode counts table for cohort
        :param covariate_df: covariate table for cohort
        :param phecode_df: phecode mapping table
        :param var_cols: variable columns in general case
        :param gender_specific_var_cols: variable columns in gender-specific case
        :param keep_ids: if True, keep phecode person_id column, used when person_id is needed
        :return: cases, controls and analysis_var_cols
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
            # filter cohort for just sex of phecode
            if sex_restriction == "Male":
                covariate_df = covariate_df.filter(pl.col(sex_at_birth_col) == male_value)
            elif sex_restriction == "Female":
                covariate_df = covariate_df.filter(pl.col(sex_at_birth_col) == female_value)
        else:
            # if data has single sex and different from sex of phecode, return empty dfs
            # otherwise, data is fine as is, i.e., nothing needed to be done
            if (
                (sex_restriction == "Male" and male_value not in sex_values)
                or (sex_restriction == "Male") and (male_value not in sex_values)
            ):
                return pl.DataFrame(), pl.DataFrame(), []

        # GENERATE CASES & CONTROLS
        if len(covariate_df) > 0:
            # CASES
            # participants with at least <min_phecode_count> phecodes
            case_ids = phecode_counts.filter(
                (pl.col("phecode") == phecode) & (pl.col("count") >= min_phecode_count)
            )["person_id"].unique().to_list()
            cases = covariate_df.filter(pl.col("person_id").is_in(case_ids))

            # CONTROLS
            # phecode exclusions
            if use_exclusion:
                exclude_range = [phecode] + self._exclude_range(phecode, phecode_df=phecode_df)
            else:
                exclude_range = [phecode]
            # get participants ids to exclude and filter covariate df
            exclude_ids = phecode_counts.filter(
                pl.col("phecode").is_in(exclude_range)
            )["person_id"].unique().to_list()
            controls = covariate_df.filter(~(pl.col("person_id").is_in(exclude_ids)))

            # PROCESS OBSERVED TIME FOR COX REGRESSION
            if method == "cox":
                # CASES
                case_observed_time_df = phecode_counts.filter(
                    pl.col("person_id").is_in(case_ids)
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

            # DUPLICATE CHECK
            # drop duplicates
            duplicate_check_cols = ["person_id"] + analysis_var_cols
            cases = cases.unique(subset=duplicate_check_cols)
            controls = controls.unique(subset=duplicate_check_cols)

            # KEEP ONLY REQUIRED COLUMNS
            if method == "cox":
                analysis_var_cols = analysis_var_cols + ["observed_time"]
                if ((sex_restriction == "Both") and
                    ((cox_stratification_col is not None) and
                     (cox_stratification_col not in analysis_var_cols))):
                    analysis_var_cols = analysis_var_cols + [cox_stratification_col]

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

    def get_phecode_data(self, phecode):
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

    @staticmethod
    def _logit_result_prep(result, var_of_interest_index):
        """
        Process result from statsmodels
        :param result: logistic regression result
        :param var_of_interest_index: index of variable of interest
        :return: dictionary with key statistics
        """
        results_as_html = result.summary().tables[0].as_html()
        converged = pd.read_html(results_as_html)[0].iloc[5, 1]
        results_as_html = result.summary().tables[1].as_html()
        res = pd.read_html(results_as_html, header=0, index_col=0)[0]

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

    def _cox_result_prep(self, result, stratified_by, warning_message=None):
        """
        Process result from statsmodels
        :param result: cox regression result
        :param stratified_by: whether cox regression was stratified or not
        :param warning_message: list of warnings to include
        :return: dictionary with key statistics
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

    def _regression(self, phecode):
        """
        Logistic regression of single phecode
        :param phecode: phecode of interest
        :return: result_dict object
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

        cases, controls, analysis_var_cols = self._case_control_prep(phecode,
                                                                     phecode_counts=phecode_counts,
                                                                     covariate_df=covariate_df,
                                                                     var_cols=var_cols,
                                                                     gender_specific_var_cols=gender_specific_var_cols)

        # only run regression if number of cases > min_cases
        if (len(cases) >= min_cases) and (len(controls) > min_cases):

            if suppress_warnings:
                warnings.simplefilter("ignore")
            else:
                warnings.simplefilter("always")

            # add case/control values
            cases = cases.with_columns(pl.Series([1] * len(cases)).alias("y"))
            controls = controls.with_columns(pl.Series([0] * len(controls)).alias("y"))

            # merge cases & controls
            regressors = cases.vstack(controls)

            # convert to numpy series
            y = regressors["y"].to_numpy()

            # base result dict
            base_dict = {"phecode": phecode,
                         "cases": len(cases),
                         "controls": len(controls)}

            # OPTION 1: COX REGRESSION
            if method == "cox":
                # for cox regression warnings is always on to catch convergence status
                warnings.simplefilter("always")

                strata = None
                stratified_by = "None"
                if cox_stratification_col in regressors.columns:
                    strata = stratified_by = cox_stratification_col
                cox = CoxPHFitter()
                combined_warning = "Converged"
                try:
                    # wrap fit() in warning handler
                    captured_warnings = []
                    with warnings.catch_warnings(record=True) as w:
                        result = cox.fit(
                            df=regressors.to_pandas(),
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
                    # print(f"Convergence error for phecode {phecode}. Lowering step_size to 0.1.")
                    combined_warning = "Convergence error. step_size was lowered to 0.1 (default is 0.95)."
                    result = cox.fit(
                        df=regressors.to_pandas(),
                        event_col="y",
                        duration_col="observed_time",
                        strata=strata,
                        fit_options={"step_size": 0.1}
                    )
                except Exception as e:
                    print("Exception:", e)
                    result = None

                # process result
                if result is not None:
                    stats_dict = self._cox_result_prep(
                        result,
                        stratified_by=stratified_by,
                        warning_message=combined_warning
                    )
                    result_dict = {**base_dict, **stats_dict}

                    # choose to see results on the fly
                    if verbose:
                        print(f"Phecode {phecode} ({len(cases)} cases/{len(controls)} controls): {result_dict}\n")

                    return result_dict

            # OPTION 2: LOGISTIC REGRESSION
            if method == "logit":
                regressors = regressors[analysis_var_cols]
                # get index of variable of interest
                var_index = regressors.columns.index(independent_variable_of_interest)
                regressors = regressors.to_numpy()
                regressors = sm.tools.add_constant(regressors, prepend=False)
                logit = sm.Logit(y, regressors, missing="drop")

                # catch Singular matrix error
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
                    # process result
                    stats_dict = self._logit_result_prep(result=result, var_of_interest_index=var_index)
                    result_dict = {**base_dict, **stats_dict}  # python 3.5 or later
                    # result_dict = base_dict | stats_dict  # python 3.9 or later

                    # choose to see results on the fly
                    if verbose:
                        print(f"Phecode {phecode} ({len(cases)} cases/{len(controls)} controls): {result_dict}\n")

                    return result_dict

        else:
            if verbose:
                print(f"Phecode {phecode} ({len(cases)} cases/{len(controls)} controls):",
                      "Not enough cases or controls. Pass.\n")

    def _batch_regression(self, phecode_batch):
        """
        Run regression for a batch of phecodes
        :param phecode_batch: batch of phecodes to run regression
        :return: list of regression results
        """
        results = []
        for phecode in phecode_batch:
            result = self._regression(phecode=phecode)
            if result is not None:
                results.append(result)
        return results

    def run(self,
            parallelization=None,
            n_workers=None):
        """
        Run parallel logistic regressions
        :param parallelization: defaults to "multithreading"; other options is "serial"
        :param n_workers: maximum number of workers
        :return: PheWAS summary statistics Polars dataframe
        """
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~    Running PheWAS    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

        # assign optimal parallelization method when it is not specified
        if parallelization is None:
            if self.method == "logit":
                parallelization = "multithreading"
            elif self.method == "cox":
                parallelization = "multiprocessing"

        result_dicts = []
        if parallelization == "serial":
            for phecode in tqdm(self.phecode_list):
                result = self._regression(
                    phecode=phecode
                )
                result_dicts.append(result)

        elif parallelization == "multithreading":
            if n_workers is None:
                n_workers = round(os.cpu_count()*2/3)
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                jobs = [
                    executor.submit(
                        self._batch_regression,
                        phecode_batch
                    ) for phecode_batch in self.phecode_batch_list
                ]
                for job in tqdm(as_completed(jobs), total=len(self.phecode_batch_list)):
                    result_dicts.extend(job.result())

        elif parallelization == "multiprocessing":
            if n_workers is None:
                n_workers = os.cpu_count() - min(round(os.cpu_count()/4), 4)
            mp_context = get_context("spawn")
            with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp_context) as executor:
                jobs = [
                    executor.submit(
                        self._batch_regression,
                        phecode_batch
                    ) for phecode_batch in self.phecode_batch_list
                ]
                for job in tqdm(as_completed(jobs), total=len(self.phecode_batch_list)):
                    result_dicts.extend(job.result())

        else:
            return "Invalid parallelization method! Select \"multithreading\", \"multiprocessing\", or \"serial\"."
        result_dicts = [result for result in result_dicts if result is not None]

        if result_dicts:
            result_df = pl.from_dicts(result_dicts)
            self.results = result_df.join(self.phecode_df[["phecode", "sex",
                                                           "phecode_string",
                                                           "phecode_category"]].unique(),
                                          how="left",
                                          on="phecode").rename({"sex": "phecode_sex_restriction"})

            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~    PheWAS Completed    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

            self.tested_count = len(self.results)
            self.not_tested_count = len(self.phecode_list) - self.tested_count
            self.bonferroni = -np.log10(0.05 / self.tested_count)
            self.phecodes_above_bonferroni = self.results.filter(pl.col("neg_log_p_value") > self.bonferroni)
            self.above_bonferroni_count = len(self.phecodes_above_bonferroni)

            # save results
            self.results.write_csv(self.output_file_name)

            print("Number of participants in cohort:", self.cohort_size)
            print("Number of phecodes in cohort:", len(self.phecode_list))
            print(f"Number of phecodes having less than {self.min_cases} cases or controls:", self.not_tested_count)
            print("Number of phecodes tested:", self.tested_count)
            print(u"Suggested Bonferroni correction (-log\u2081\u2080 scale):", self.bonferroni)
            print("Number of phecodes above Bonferroni correction:", self.above_bonferroni_count)
            print()
            print("PheWAS results saved to\033[1m", self.output_file_name, "\033[0m")
        else:
            print("No analysis done. Please check your inputs.")

        print()


def main():

    # parse args
    parser = argparse.ArgumentParser(description="PheWAS analysis tool.")
    parser.add_argument("-p",
                        "--phecode_count_csv_path",
                        type=str, required=True,
                        help="Path to the phecode count csv file.")
    parser.add_argument("-c",
                        "--cohort_csv_path",
                        type=str, required=True,
                        help="Path to the cohort csv file.")
    parser.add_argument("-pv",
                        "--phecode_version",
                        type=str, required=True, choices=["1.2", "X"],
                        help="Phecode version.")
    parser.add_argument("-m",
                        "--method",
                        type=str, required=False, choices=["logit", "cox"],
                        help="Phecode regression method. Can be 'logit' or 'cox'.")
    parser.add_argument("--cox_control_observed_time_col",
                        type=str, required=False,
                        help="Observed time for controls in phecode regression. Right censored time.")
    parser.add_argument("--cox_phecode_observed_time_col",
                        type=str, required=False,
                        help="Observed time for cases in phecode regression. First phecode event time.")
    parser.add_argument("--cox_stratification_col",
                        type=str, required=False,
                        help="Stratification for cox regression.")
    parser.add_argument("-cv",
                        "--covariates",
                        nargs="+",
                        type=str, required=True,
                        help="List of covariates to use in PheWAS analysis.")
    parser.add_argument("-i",
                        "--independent_variable_of_interest",
                        type=str, required=True,
                        help="Independent variable of interest.")
    parser.add_argument("-s",
                        "--sex_at_birth_col",
                        type=str, required=True,
                        help="Sex at birth column.")
    parser.add_argument("-mso",
                        "--male_as_one",
                        type=bool, required=False,
                        help="Whether male was assigned as 1 in data.")
    parser.add_argument("-pl",
                        "--phecode_to_process",
                        nargs="+",
                        type=str, required=False, default="all",
                        help="List of specific phecodes to use in PheWAS analysis.")
    parser.add_argument("-e",
                        "--use_exclusion",
                        type=bool, required=False, default=False,
                        help="Whether to use phecode exclusions. Only applicable for phecode 1.2.")
    parser.add_argument("-mc",
                        "--min_case",
                        type=int, required=False, default=50,
                        help="Minimum number of cases required to be tested.")
    parser.add_argument("-mpc",
                        "--min_phecode_count",
                        type=int, required=False, default=2,
                        help="Minimum number of phecode counts required to be considered as case.")
    parser.add_argument("--n_workers",
                        type=int, required=False, default=round(os.cpu_count()*2/3),
                        help="Number of threads to use for parallel.")
    parser.add_argument("-o",
                        "--output_file_name",
                        type=str, required=False, default="phewas_results.csv")
    parser.add_argument("--parallelization",
                        type=str, required=False, default="multithreading")
    parser.add_argument("--batch_size",
                        type=int, required=False, default=10, help="Batch size for parallelization.")
    parser.add_argument("--suppress_warnings",
                        type=bool, required=False, default=True, help="Whether to suppress warnings.")
    args = parser.parse_args()

    # run PheWAS
    phewas = PheWAS(phecode_version=args.phecode_version,
                    phecode_count_csv_path=args.phecode_count_csv_path,
                    cohort_csv_path=args.cohort_csv_path,
                    sex_at_birth_col=args.sex_at_birth_col,
                    male_as_one=args.male_as_one,
                    covariate_cols=args.covariates,
                    independent_variable_of_interest=args.independent_variable_of_interest,
                    cox_control_observed_time_col=args.cox_control_observed_time_col,
                    cox_phecode_observed_time_col=args.cox_phecode_observed_time_col,
                    cox_stratification_col=args.cox_stratification_col,
                    phecode_to_process=args.phecode_to_process,
                    use_exclusion=args.use_exclusion,
                    min_cases=args.min_case,
                    min_phecode_count=args.min_phecode_count,
                    output_file_name=args.output_file_name,
                    method=args.method,
                    batch_size=args.batch_size,
                    suppress_warnings=args.suppress_warnings)
    phewas.run(parallelization=args.parallelization,
               n_workers=args.n_workers)


if __name__ == "__main__":
    main()
