from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
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
                 sex_at_birth_col,
                 covariate_cols,
                 independent_variable_of_interest,
                 male_as_one=True,
                 icd_version="US",
                 phecode_map_file_path=None,
                 phecode_to_process="all",
                 min_cases=50,
                 min_phecode_count=2,
                 use_exclusion=False,
                 output_file_name=None,
                 verbose=False,
                 suppress_warnings=True):
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
        self.phecode_counts = pl.read_csv(phecode_count_csv_path,
                                          dtypes={"phecode": str})

        # load covariate data
        # make sure person_id in covariate data has the same type as person_id in phecode count
        self.covariate_df = pl.read_csv(cohort_csv_path)
        self.covariate_df = self.covariate_df.drop_nulls()

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

    def _exclude_range(self, phecode, phecode_df=None):
        """
        Process text data in exclude_range column; exclusively for phecodeX
        :param phecode: phecode of interest
        :return: processed exclude_range, either None or a valid list of phecode(s)
        """
        if phecode_df is None:
            phecode_df = self.phecode_df.clone()

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
                           var_cols=None, gender_specific_var_cols=None):
        """
        :param phecode: phecode of interest
        :param phecode_counts: phecode counts table for cohort
        :param covariate_df: covariate table for cohort
        :param phecode_df: phecode mapping table
        :param var_cols: variable columns in general case
        :param gender_specific_var_cols: variable columns in gender-specific case
        :return: cases, controls and analysis_var_cols
        """
        if phecode_counts is None:
            phecode_counts = self.phecode_counts.clone()
        if covariate_df is None:
            covariate_df = self.covariate_df.clone()
        if phecode_df is None:
            phecode_df = self.phecode_df.clone()
        if var_cols is None:
            var_cols = copy.deepcopy(self.var_cols)
        if gender_specific_var_cols is None:
            gender_specific_var_cols = copy.deepcopy(self.gender_specific_var_cols)

        # SEX RESTRICTION
        filtered_phecode_df = phecode_df.filter(pl.col("phecode") == phecode)
        if len(filtered_phecode_df["sex"].unique().to_list()) > 0:
            sex_restriction = filtered_phecode_df["sex"].unique().to_list()[0]
        else:
            return pl.DataFrame(), pl.DataFrame(), []

        # ANALYSIS VAR COLS
        if sex_restriction == "Both" and self.sex_as_covariate:
            analysis_var_cols = var_cols
        else:
            analysis_var_cols = gender_specific_var_cols

        # FILTER COVARIATE DATA BY PHECODE SEX RESTRICTION
        if not self.data_has_single_sex:
            if sex_restriction == "Male":
                covariate_df = covariate_df.filter(pl.col(self.sex_at_birth_col) == self.male_value)
            elif sex_restriction == "Female":
                covariate_df = covariate_df.filter(pl.col(self.sex_at_birth_col) == self.female_value)
        else:
            if (
                (sex_restriction == "Male" and self.male_value not in self.sex_values)
                or (sex_restriction == "Male") and (self.male_value not in self.sex_values)
            ):
                return pl.DataFrame(), pl.DataFrame(), []

        # GENERATE CASES & CONTROLS
        if len(covariate_df) > 0:
            # CASE
            # participants with at least <min_phecode_count> phecodes
            case_ids = phecode_counts.filter(
                (pl.col("phecode") == phecode) & (pl.col("count") >= self.min_phecode_count)
            )["person_id"].unique().to_list()
            cases = covariate_df.filter(pl.col("person_id").is_in(case_ids))

            # CONTROLS
            # phecode exclusions
            if self.use_exclusion:
                exclude_range = [phecode] + self._exclude_range(phecode, phecode_df=phecode_df)
            else:
                exclude_range = [phecode]
            # get participants ids to exclude and filter covariate df
            exclude_ids = phecode_counts.filter(
                pl.col("phecode").is_in(exclude_range)
            )["person_id"].unique().to_list()
            controls = covariate_df.filter(~(pl.col("person_id").is_in(exclude_ids)))

            # DUPLICATE CHECK
            # drop duplicates and keep analysis covariate cols only
            duplicate_check_cols = ["person_id"] + analysis_var_cols
            cases = cases.unique(subset=duplicate_check_cols)[analysis_var_cols]
            controls = controls.unique(subset=duplicate_check_cols)[analysis_var_cols]

            # KEEP ONLY REQUIRED COLUMNS
            cases = cases[analysis_var_cols]
            controls = controls[analysis_var_cols]

            return cases, controls, analysis_var_cols

        else:
            return pl.DataFrame(), pl.DataFrame(), []

    @staticmethod
    def _result_prep(result, var_of_interest_index):
        """
        Process result from statsmodels
        :param result: logistic regression result
        :param var_of_interest_index: index of variable of interest
        :return: dataframe with key statistics
        """
        results_as_html = result.summary().tables[0].as_html()
        converged = pd.read_html(results_as_html)[0].iloc[5, 1]
        results_as_html = result.summary().tables[1].as_html()
        res = pd.read_html(results_as_html, header=0, index_col=0)[0]

        p_value = result.pvalues[var_of_interest_index]
        neg_log_p_value = -np.log10(p_value)
        beta = result.params[var_of_interest_index]
        conf_int_1 = res.iloc[var_of_interest_index]['[0.025']
        conf_int_2 = res.iloc[var_of_interest_index]['0.975]']
        odds_ratio = np.exp(beta)
        log10_odds_ratio = np.log10(odds_ratio)

        return {"p_value": p_value,
                "neg_log_p_value": neg_log_p_value,
                "beta": beta,
                "conf_int_1": conf_int_1,
                "conf_int_2": conf_int_2,
                "odds_ratio": odds_ratio,
                "log10_odds_ratio": log10_odds_ratio,
                "converged": converged}

    def _logistic_regression(self, phecode,
                             phecode_counts=None, covariate_df=None,
                             var_cols=None, gender_specific_var_cols=None):
        """
        Logistic regression of single phecode
        :param phecode: phecode of interest
        :param phecode_counts: phecode counts table for cohort
        :param covariate_df: covariate table for cohort
        :param var_cols: variable columns in general case
        :param gender_specific_var_cols: variable columns in gender-specific case
        :return: result_dict object
        """

        if phecode_counts is None:
            phecode_counts = self.phecode_counts.clone()
        if covariate_df is None:
            covariate_df = self.covariate_df.clone()
        if var_cols is None:
            var_cols = copy.deepcopy(self.var_cols)
        if gender_specific_var_cols is None:
            gender_specific_var_cols = copy.deepcopy(self.gender_specific_var_cols)

        cases, controls, analysis_var_cols = self._case_control_prep(phecode,
                                                                     phecode_counts=phecode_counts,
                                                                     covariate_df=covariate_df,
                                                                     var_cols=var_cols,
                                                                     gender_specific_var_cols=gender_specific_var_cols)

        # only run regression if number of cases > min_cases
        if (len(cases) >= self.min_cases) and (len(controls) > self.min_cases):

            # add case/control values
            cases = cases.with_columns(pl.Series([1] * len(cases)).alias("y"))
            controls = controls.with_columns(pl.Series([0] * len(controls)).alias("y"))

            # merge cases & controls
            regressors = cases.vstack(controls)

            # get index of variable of interest
            var_index = regressors[analysis_var_cols].columns.index(self.independent_variable_of_interest)

            # logistic regression
            if self.suppress_warnings:
                warnings.simplefilter("ignore")
            y = regressors["y"].to_numpy()
            regressors = regressors[analysis_var_cols].to_numpy()
            regressors = sm.tools.add_constant(regressors, prepend=False)
            logit = sm.Logit(y, regressors, missing="drop")

            # catch Singular matrix error
            try:
                result = logit.fit(disp=False)
            except (np.linalg.linalg.LinAlgError, statsmodels.tools.sm_exceptions.PerfectSeparationError) as err:
                if "Singular matrix" in str(err) or "Perfect separation" in str(err):
                    if self.verbose:
                        print(f"Phecode {phecode} ({len(cases)} cases/{len(controls)} controls):", str(err), "\n")
                    pass
                else:
                    raise
                result = None

            if result is not None:
                # process result
                base_dict = {"phecode": phecode,
                             "cases": len(cases),
                             "controls": len(controls)}
                stats_dict = self._result_prep(result=result, var_of_interest_index=var_index)
                result_dict = {**base_dict, **stats_dict}  # python 3.5 or later
                # result_dict = base_dict | stats_dict  # python 3.9 or later

                # choose to see results on the fly
                if self.verbose:
                    print(f"Phecode {phecode} ({len(cases)} cases/{len(controls)} controls): {result_dict}\n")

                return result_dict

        else:
            if self.verbose:
                print(f"Phecode {phecode} ({len(cases)} cases/{len(controls)} controls):",
                      "Not enough cases or controls. Pass.\n")

    def run(self,
            parallelization="multithreading",
            n_threads=round(os.cpu_count()*2/3)):
        """
        Run parallel logistic regressions
        :param parallelization: defaults to "multithreading", utilizing concurrent.futures.ThreadPoolExecutor();
                                if "multiprocessing": use multiprocessing.Pool()
        :param n_threads: number of threads in multithreading
        :return: PheWAS summary statistics Polars dataframe
        """
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~    Running PheWAS    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

        if parallelization == "multithreading":
            with ThreadPoolExecutor(max_workers=n_threads) as executor:
                jobs = [
                    executor.submit(
                        self._logistic_regression,
                        phecode,
                        self.phecode_counts.clone(),
                        self.covariate_df.clone(),
                        copy.deepcopy(self.var_cols),
                        copy.deepcopy(self.gender_specific_var_cols)
                    ) for phecode in self.phecode_list
                ]
                result_dicts = [job.result() for job in tqdm(as_completed(jobs), total=len(self.phecode_list))]
        else:
            return "Invalid parallelization method! Currently only supports \"multithreading\""
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
                        type=str, required=True, choices=["1.2", 'X'],
                        help="Phecode version.")
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
    parser.add_argument("-t",
                        "--threads",
                        type=int, required=False, default=round(os.cpu_count()*2/3),
                        help="Number of threads to use for parallel.")
    parser.add_argument("-o",
                        "--output_file_name",
                        type=str, required=False, default="phewas_results.csv")
    args = parser.parse_args()

    # run PheWAS
    phewas = PheWAS(phecode_version=args.phecode_version,
                    phecode_count_csv_path=args.phecode_count_csv_path,
                    cohort_csv_path=args.cohort_csv_path,
                    sex_at_birth_col=args.sex_at_birth_col,
                    male_as_one=args.male_as_one,
                    covariate_cols=args.covariates,
                    independent_variable_of_interest=args.independent_variable_of_interest,
                    phecode_to_process=args.phecode_to_process,
                    use_exclusion=args.use_exclusion,
                    min_cases=args.min_case,
                    min_phecode_count=args.min_phecode_count,
                    output_file_name=args.output_file_name)
    phewas.run(n_threads=args.threads)


if __name__ == "__main__":
    main()
