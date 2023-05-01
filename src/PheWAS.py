from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import multiprocessing
import numpy as np
import pandas as pd
import polars as pl
import statsmodels.api as sm
import warnings


class PheWAS:

    def __init__(self,
                 phecode_df,
                 phecode_counts,
                 covariate_df,
                 gender_col,
                 covariate_cols,
                 independent_var_col,
                 phecode_to_process="all",
                 min_cases=50,
                 min_phecode_count=2,
                 use_exclusion=True,
                 verbose=False,
                 suppress_warnings=True):
        """
        :param phecode_df: dataframe contains phecode information & mapping; included in folder "phecode"
        :param phecode_counts: phecode count of relevant participants at minimum
        :param covariate_df: dataframe contains person_id and covariates of interest;
                             must have both "male" and "female" columns
        :param gender_col: gender/sex column of interest; either "male" or "female" column can be used
        :param covariate_cols: name of covariate columns; excluding independent var of interest
        :param independent_var_col: binary "case" column to specify participants with/without variant of interest
        :param phecode_to_process: defaults to "all"; otherwise, a list of phecodes must be provided
        :param min_cases: defaults to 50; minimum number of cases for each phecode to be considered for PheWAS
        :param min_phecode_count: defaults to 2; minimum number of phecode count to qualify as case for PheWAS
        :param use_exclusion: defaults to True; whether to use additional exclusion range in control for PheWAS
        :param verbose: defaults to False; if True, print brief result of each phecode run
        :param suppress_warnings: defaults to True;
                                  if True, ignore common exception warnings such as ConvergenceWarnings, etc.
        """

        print("~~~~~~~~~~~~~~~~~~~~~~~~    Creating PheWAS Object    ~~~~~~~~~~~~~~~~~~~~~~~~~~")

        # basic attributes from instantiation
        self.phecode_df = phecode_df
        self.phecode_counts = self._to_polars(phecode_counts)
        self.covariate_df = self._to_polars(covariate_df)
        self.gender_col = gender_col
        self.covariate_cols = covariate_cols
        self.independent_var_col = independent_var_col
        self.verbose = verbose
        self.min_cases = min_cases
        self.min_phecode_count = min_phecode_count
        self.use_exclusion = use_exclusion
        self.suppress_warnings = suppress_warnings

        # additional attributes
        self.gender_specific_var_cols = [self.independent_var_col] + self.covariate_cols
        self.var_cols = [self.independent_var_col] + self.covariate_cols + [self.gender_col]
        self.merged_df = covariate_df.join(phecode_counts, how="inner", on="person_id")
        self.cohort_size = covariate_df.n_unique()
        if phecode_to_process == "all":
            self.phecode_list = self.merged_df["phecode"].unique().to_list()
        else:
            self.phecode_list = phecode_to_process

        # attributes for reporting PheWAS results
        self.result = None
        self.not_tested_count = 0
        self.tested_count = 0
        self.bonferroni = None
        self.phecodes_above_bonferroni = None
        self.above_bonferroni_count = None

    @staticmethod
    def _to_polars(df):
        """
        check and convert pandas dataframe object to polars dataframe, if applicable
        :param df: dataframe object
        :return: polars dataframe
        """
        if isinstance(df, pd.DataFrame):
            return pl.from_pandas(df)
        else:
            pass

    def _sex_restriction(self, phecode):
        """
        :param phecode: phecode of interest
        :return: sex restriction and respective analysis covariates
        """

        filtered_df = self.phecode_df.filter(pl.col("phecode") == phecode)
        sex_restriction = filtered_df["sex"].unique().to_list()[0]

        if sex_restriction == "Both":
            analysis_var_cols = self.var_cols
        else:
            analysis_var_cols = self.gender_specific_var_cols

        return sex_restriction, analysis_var_cols

    def _exclude_range(self, phecode):
        """
        process text data in exclude_range column; exclusively for phecodeX
        :param phecode: phecode of interest
        :return: processed exclude_range, either None or a valid list of phecode(s)
        """
        # not all phecode has exclude_range
        # exclude_range can be single code (e.g., "777"), single range (e.g., "777-780"),
        # or multiple ranges/codes (e.g., "750-777,586.2")
        phecodes_without_exclude_range = self.phecode_df.filter(
            pl.col("exclude_range").is_null()
        )["phecode"].unique().to_list()
        if phecode in phecodes_without_exclude_range:
            exclude_range = []
        else:
            # get exclude_range value of phecode
            ex_val = self.phecode_df.filter(pl.col("phecode") == phecode)["exclude_range"].unique().to_list()[0]

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

    def _case_prep(self, phecode):
        """
        prepare PheWAS case data
        :param phecode: phecode of interest
        :return: polars dataframe of case data
        """

        # case participants with at least <min_phecode_count> phecodes
        cases = self.merged_df.filter((pl.col("phecode") == phecode) &
                                      (pl.col("count") >= self.min_phecode_count))

        # select data based on phecode "sex", e.g., male/female only or both
        sex_restriction, analysis_var_cols = self._sex_restriction(phecode)
        if sex_restriction == "Male":
            cases = cases.filter(pl.col("male") == 1)
        elif sex_restriction == "Female":
            cases = cases.filter(pl.col("female") == 1)

        # drop duplicates and keep analysis covariate cols only
        duplicate_check_cols = ["person_id"] + analysis_var_cols
        cases = cases.unique(subset=duplicate_check_cols)[analysis_var_cols]

        return cases

    def _control_prep(self, phecode):
        """
        prepare PheWAS control data
        :param phecode: phecode of interest
        :return: polars dataframe of control data
        """

        # phecode exclusions
        if self.use_exclusion:
            exclude_range = [phecode] + self._exclude_range(phecode)
        else:
            exclude_range = [phecode]

        # select control data based on
        sex_restriction, analysis_covariate_cols = self._sex_restriction(phecode)
        base_controls = self.merged_df.filter(~(pl.col("phecode").is_in(exclude_range)))
        if sex_restriction == "Male":
            controls = base_controls.filter(pl.col("male") == 1)
        elif sex_restriction == "Female":
            controls = base_controls.filter(pl.col("female") == 1)
        else:
            controls = base_controls

        # drop duplicates and keep analysis covariate cols only
        duplicate_check_cols = ["person_id"] + analysis_covariate_cols
        controls = controls.unique(subset=duplicate_check_cols)[analysis_covariate_cols]

        return controls

    @staticmethod
    def _result_prep(result, var_of_interest_index):
        """
        process result from statsmodels
        :param result: logistic regression result
        :param var_of_interest_index: index of variable of interest
        :return: dataframe with key statistics
        """

        results_as_html = result.summary().tables[0].as_html()
        converged = pd.read_html(results_as_html)[0].iloc[5, 1]
        results_as_html = result.summary().tables[1].as_html()
        res = pd.read_html(results_as_html, header=0, index_col=0)[0]

        p_value = result.pvalues[var_of_interest_index]
        beta_ind = result.params[var_of_interest_index]
        conf_int_1 = res.iloc[var_of_interest_index]['[0.025']
        conf_int_2 = res.iloc[var_of_interest_index]['0.975]']
        neg_log_p_value = -np.log10(p_value)

        return {"p_value": p_value,
                "neg_log_p_value": neg_log_p_value,
                "beta_ind": beta_ind,
                "conf_int_1": conf_int_1,
                "conf_int_2": conf_int_2,
                "converged": converged}

    def _logistic_regression(self, phecode):
        """
        logistic regression of single phecode
        :param phecode: phecode of interest
        :return: logistic regression result object
        """

        sex_restriction, analysis_covariate_cols = self._sex_restriction(phecode)
        cases = self._case_prep(phecode)

        # only run regression if number of cases > min_cases
        if len(cases) >= self.min_cases:
            controls = self._control_prep(phecode)

            # add case/control values
            cases = cases.with_columns(pl.Series([1] * len(cases)).alias("y"))
            controls = controls.with_columns(pl.Series([0] * len(controls)).alias("y"))

            # merge cases & controls
            regressors = cases.vstack(controls)

            # get index of independent_var_col
            var_index = regressors[analysis_covariate_cols].columns.index(self.independent_var_col)

            # logistic regression
            if self.suppress_warnings:
                warnings.simplefilter("ignore")
            y = regressors["y"].to_numpy()
            regressors = regressors[analysis_covariate_cols].to_numpy()
            regressors = sm.tools.add_constant(regressors, prepend=False)
            logit = sm.Logit(y, regressors, missing="drop")
            result = logit.fit(disp=False)

            # process result
            base_dict = {"phecode": phecode,
                         "cases": len(cases),
                         "controls": len(controls)}
            stats_dict = self._result_prep(result=result, var_of_interest_index=var_index)
            result_dict = base_dict | stats_dict

            # choose to see results on the fly
            if self.verbose:
                print(f"Phecode {phecode}: {result_dict}")

            return result_dict

        else:
            if self.verbose:
                print(f"Phecode {phecode}: {len(cases)} cases - Not enough cases. Pass.")

    # now define function for running PheWAS
    def run(self, multi_threaded=True):

        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~    Running PheWAS   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

        result_dicts = []
        if multi_threaded:
            with ThreadPoolExecutor() as executor:
                jobs = [executor.submit(self._logistic_regression, phecode) for phecode in self.phecode_list]
                for job in tqdm(as_completed(jobs), total=len(self.phecode_list)):
                    try:
                        result = job.result()
                    except np.linalg.linalg.LinAlgError as err:
                        if "Singular matrix" in str(err):
                            pass
                        else:
                            raise
                    if result:
                        result_dicts.append(result)
        else:
            with multiprocessing.Pool(multiprocessing.cpu_count()-1) as p:
                result_dicts = list(tqdm(p.imap(self._logistic_regression, self.phecode_list),
                                                total=len(self.phecode_list)))
        result_dicts = [result for result in result_dicts if result]
        result_df = pl.from_dicts(result_dicts)
        self.result = result_df.join(self.phecode_df[["phecode", "phecode_string", "phecode_category"]].unique(),
                                     how="left",
                                     on="phecode")

        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~    PheWAS Completed    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

        self.tested_count = len(self.result)
        self.not_tested_count = len(self.phecode_list) - self.tested_count
        self.bonferroni = -np.log10(0.05 / self.tested_count)
        self.phecodes_above_bonferroni = self.result.filter(pl.col("neg_log_p_value") > self.bonferroni)
        self.above_bonferroni_count = len(self.phecodes_above_bonferroni)

        print("Number of participants in cohort:", self.cohort_size)
        print("Number of phecodes in cohort:", len(self.phecode_list))
        print(f"Number of phecodes having less than {self.min_cases} cases:", self.not_tested_count)
        print("Number of phecodes tested:", self.tested_count)
        print(u"Suggested Bonferroni correction (-log\u2081\u2080 scale):", self.bonferroni)
        print("Number of phecodes above Bonferroni correction:", self.above_bonferroni_count)
