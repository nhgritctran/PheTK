from . import genotype, covariate
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.notebook import tqdm
import copy
import numpy as np
import os
import pandas as pd
import polars as pl
import statsmodels.api as sm
import sys
import time
import warnings


class Cohort:

    def __init__(self,
                 db,
                 db_version):
        self.db = db
        self.db_version = db_version
        self.genotype_cohort = None
        self.final_cohort = None

    def by_genotype(self,
                    chromosome_number,
                    genomic_position,
                    ref_allele,
                    alt_allele,
                    case_gt,
                    control_gt,
                    reference_genome="GRCh38",
                    mt_path=None,
                    output_file_name=None):
        self.genotype_cohort = genotype.build_variant_cohort(chromosome_number=chromosome_number,
                                                             genomic_position=genomic_position,
                                                             ref_allele=ref_allele,
                                                             alt_allele=alt_allele,
                                                             case_gt=case_gt,
                                                             control_gt=control_gt,
                                                             reference_genome=reference_genome,
                                                             db=self.db,
                                                             db_version=self.db_version,
                                                             mt_path=mt_path,
                                                             output_file_name=output_file_name)

    def add_covariates(self,
                       cohort=None,
                       natural_age=True,
                       age_at_last_event=True,
                       sex_at_birth=True,
                       ehr_length=True,
                       dx_code_count=True,
                       genetic_ancestry=False,
                       first_n_pcs=0,
                       chunk_size=10000,
                       drop_nulls=True):
        if cohort is not None:
            pass
        elif cohort is None and self.genotype_cohort is not None:
            cohort = self.genotype_cohort
        else:
            print("A cohort is required.")
            sys.exit(0)
        participant_ids = cohort["person_id"].unique().to_list()
        covariates = covariate.get_covariates(participant_ids=participant_ids,
                                              natural_age=natural_age,
                                              age_at_last_event=age_at_last_event,
                                              sex_at_birth=sex_at_birth,
                                              ehr_length=ehr_length,
                                              dx_code_count=dx_code_count,
                                              genetic_ancestry=genetic_ancestry,
                                              first_n_pcs=first_n_pcs,
                                              db_version=self.db_version,
                                              chunk_size=chunk_size)
        self.final_cohort = cohort.join(covariates, how="left", on="person_id")
        if drop_nulls:
            self.final_cohort = self.final_cohort.drop_nulls()


class PheWAS:

    def __init__(self,
                 phecode_version,
                 phecode_count_csv_path,
                 cohort_csv_path,
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
        :param phecode_version: accepts "1.2" or "X"
        :param phecode_count_csv_path: path to phecode count of relevant participants at minimum
        :param cohort_csv_path: path to cohort data with covariates of interest
        :param gender_col: gender/sex column of interest, by default, male = 1, female = 0
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

        # load phecode mapping file
        cwd = os.getcwd()
        if phecode_version == "X":
            # noinspection PyTypeChecker
            self.phecode_df = pl.read_csv(f"{cwd}/PyPheWAS/phecode/phecodeX.csv",
                                          dtypes={"phecode": str,
                                                  "ICD": str,
                                                  "exclude_range": str,
                                                  "phecode_top": str})
        elif phecode_version == "1.2":
            # noinspection PyTypeChecker
            self.phecode_df = pl.read_csv(f"{cwd}/PyPheWAS/phecode/phecode12.csv",
                                          dtypes={"phecode": str,
                                                  "ICD": str,
                                                  "exclude_range": str,
                                                  "phecode_top": str})
        else:
            print("Unsupported phecode version. Supports phecode \"1.2\" and \"X\".")
            sys.exit(0)

        # load phecode counts data for all participants
        # noinspection PyTypeChecker
        self.phecode_counts = pl.read_csv(phecode_count_csv_path,
                                          dtypes={"phecode": str})

        # load covariate data
        # make sure person_id in covariate data has the same type as person_id in phecode count
        self.covariate_df = pl.read_csv(cohort_csv_path)

        # basic attributes from instantiation
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

        # check for string type variables among covariates
        if pl.Utf8 in self.covariate_df[self.var_cols].schema.values():
            str_cols = [k for k, v in self.covariate_df.schema.items() if v is pl.Utf8]
            print(f"Column(s) {str_cols} contain(s) string type. Only numerical types are accepted.")
            sys.exit(0)

        # keep only relevant columns in covariate_df
        cols_to_keep = list(set(["person_id"] + self.var_cols))
        self.covariate_df = self.covariate_df[cols_to_keep]
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
            polars_df = pl.from_pandas(df)
        else:
            polars_df = df
        return polars_df

    def _exclude_range(self, phecode, phecode_df=None):
        """
        process text data in exclude_range column; exclusively for phecodeX
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
        filtered_df = phecode_df.filter(pl.col("phecode") == phecode)
        sex_restriction = filtered_df["sex"].unique().to_list()[0]

        # ANALYSIS VAR COLS
        if sex_restriction == "Both":
            analysis_var_cols = var_cols
        else:
            analysis_var_cols = gender_specific_var_cols

        # CASE
        # participants with at least <min_phecode_count> phecodes
        case_ids = phecode_counts.filter(
            (pl.col("phecode") == phecode) & (pl.col("count") >= self.min_phecode_count)
        )["person_id"].unique().to_list()
        cases = covariate_df.filter(pl.col("person_id").is_in(case_ids))
        # select data based on phecode "sex", e.g., male/female only or both
        if sex_restriction == "Male":
            cases = cases.filter(pl.col(self.gender_col) == 1)
        elif sex_restriction == "Female":
            cases = cases.filter(pl.col(self.gender_col) == 0)

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
        base_controls = covariate_df.filter(~(pl.col("person_id").is_in(exclude_ids)))
        if sex_restriction == "Male":
            controls = base_controls.filter(pl.col(self.gender_col) == 1)
        elif sex_restriction == "Female":
            controls = base_controls.filter(pl.col(self.gender_col) == 0)
        else:
            controls = base_controls

        # DUPLICATE CHECK
        # drop duplicates and keep analysis covariate cols only
        duplicate_check_cols = ["person_id"] + analysis_var_cols
        cases = cases.unique(subset=duplicate_check_cols)[analysis_var_cols]
        controls = controls.unique(subset=duplicate_check_cols)[analysis_var_cols]

        # KEEP MINIMUM REQUIRED COLUMNS
        cases = cases[analysis_var_cols]
        controls = controls[analysis_var_cols]

        return cases, controls, analysis_var_cols

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

    def _logistic_regression(self, phecode,
                             phecode_counts=None, covariate_df=None,
                             var_cols=None, gender_specific_var_cols=None):
        """
        logistic regression of single phecode
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

        case_start_time = time.time()
        cases, controls, analysis_var_cols = self._case_control_prep(phecode,
                                                                     phecode_counts=phecode_counts,
                                                                     covariate_df=covariate_df,
                                                                     var_cols=var_cols,
                                                                     gender_specific_var_cols=gender_specific_var_cols)
        case_end_time = time.time()
        if self.verbose:
            print(f"Phecode {phecode} cases & controls created in {case_end_time - case_start_time} seconds\n")

        # only run regression if number of cases > min_cases
        if len(cases) >= self.min_cases:

            # add case/control values
            cases = cases.with_columns(pl.Series([1] * len(cases)).alias("y"))
            controls = controls.with_columns(pl.Series([0] * len(controls)).alias("y"))

            # merge cases & controls
            regressors = cases.vstack(controls)

            # get index of independent_var_col
            var_index = regressors[analysis_var_cols].columns.index(self.independent_var_col)

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
            except np.linalg.linalg.LinAlgError as err:
                if "Singular matrix" in str(err):
                    pass
                else:
                    raise
                result = None

            if result:
                # process result
                base_dict = {"phecode": phecode,
                             "cases": len(cases),
                             "controls": len(controls)}
                stats_dict = self._result_prep(result=result, var_of_interest_index=var_index)
                result_dict = {**base_dict, **stats_dict}  # python 3.5 or later
                # result_dict = base_dict | stats_dict  # python 3.9 or later

                # choose to see results on the fly
                if self.verbose:
                    print(f"Phecode {phecode}: {result_dict}\n")

                return result_dict

        else:
            if self.verbose:
                print(f"Phecode {phecode}: {len(cases)} cases - Not enough cases. Pass.\n")

    def run(self,
            parallelization="multithreading",
            n_threads=None):
        """
        run parallel logistic regressions
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
        # # WIP for multiprocessing, though multithread is faster
        # elif parallelization == "multiprocessing":
        #     with multiprocessing.Pool(min(n_cores, multiprocessing.cpu_count()-1)) as p:
        #         results = p.starmap_async(
        #             self._logistic_regression,
        #             [
        #                 (
        #                     phecode,
        #                     self.phecode_counts.clone(),
        #                     self.covariate_df.clone(),
        #                     copy.deepcopy(self.var_cols),
        #                     copy.deepcopy(self.gender_specific_var_cols)
        #                 ) for phecode in self.phecode_list
        #             ]
        #         )
        #         result_dicts = [result for result in tqdm(results.get(), total=len(self.phecode_list))]
        else:
            return "Invalid parallelization method! Currently only supports \"multithreading\""
        result_dicts = [result for result in result_dicts if result is not None]
        if result_dicts:
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
        else:
            print("No analysis done.")
