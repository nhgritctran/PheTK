import multiprocessing
import pandas as pd
import polars as pl
import statsmodels.api as sm


class PheWAS:

    def __init__(self,
                 phecode_df,
                 phecode_counts,
                 covariate_df,
                 gender_col=None,
                 covariate_cols=None,
                 independent_var_col=None,
                 phecode_to_process="all",
                 min_cases=50,
                 min_phecode_count=2,
                 verbose=False):

        print("~~~~~~~~~~~~~~~~~~~~~~~    Creating PheWAS Object    ~~~~~~~~~~~~~~~~~~~~~~~~~")

        # basic attributes
        self.phecode_df = phecode_df
        self.phecode_counts = self._to_polars(phecode_counts)
        self.covariate_df = self._to_polars(covariate_df)
        self.gender_col = gender_col
        self.covariate_cols = covariate_cols
        self.independent_var_col = independent_var_col
        self.verbose = verbose
        self.min_cases = min_cases
        self.min_phecode_count = min_phecode_count
        self.cores = multiprocessing.cpu_count() - 1

        # self.phecode_list
        if phecode_to_process == "all":
            self.phecode_list = phecode_counts["phecode"].unique().tolist()
        else:
            self.phecode_list = phecode_to_process

        # merge phecode_counts and covariate_df and define column name groups
        self.merged_df = pl.join(covariate_df, phecode_counts, how="inner", on="person_id")
        self.covariate_cols = self.independent_var_col + self.covariate_cols + self.gender_col
        self.gender_specific_covariate_cols = self.independent_var_col + self.covariate_cols

        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    Done    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

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
            analysis_covariate_cols = self.covariate_cols
        else:
            analysis_covariate_cols = self.gender_specific_covariate_cols

        return sex_restriction, analysis_covariate_cols

    def _exclude_range(self, phecode):
        """
        process text data in exclude_range column; exclusively for phecodeX
        :param phecode: phecode of interest
        :return: processed exclude_range, either None or a valid list of phecode(s)
        """
        # not all phecode has exclude_range
        # exclude_range can be single code (e.g., "777"), single range (e.g., "777-780"),
        # or multiple ranges/codes (e.g., "750-777,586.2")
        phecodes_without_exclude_range = self.merged_df.filter(
            pl.col("exclude_range").is_null()["phecode"].unique().to_list()
        )
        if phecode in phecodes_without_exclude_range:
            exclude_range = []
        else:
            # get exclude_range value of phecode
            ex_val = self.merged_df.filter(pl.col("phecode") == phecode)["exclude_range"].unique().to_list()[0]

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

        # case participants with at least 2 of the phecode
        cases = self.merged_df.filter((pl.col("phecode") == phecode) &
                                      (pl.col("count") >= self.min_phecode_count))

        # select data based on phecode "sex", e.g., male/female only or both
        sex_restriction, analysis_covariate_cols = self._sex_restriction(phecode)
        if sex_restriction == "Male":
            cases = cases.filter(pl.col("male") == 1)
        elif sex_restriction == "Female":
            cases = cases.filter(pl.col("female") == 1)

        # drop duplicates and keep analysis covariate cols only
        duplicate_check_cols = ["person_id"] + analysis_covariate_cols
        cases = cases.unique(subset=duplicate_check_cols)[analysis_covariate_cols]

        return cases

    def _control_prep(self, phecode, use_exclusion=True):
        """
        prepare PheWAS control data
        :param phecode: phecode of interest
        :return: polars dataframe of control data
        """

        # phecode exclusions
        if use_exclusion:
            exclude_range = self._exclude_range(phecode) + [phecode]
        else:
            exclude_range = []

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

    def result_prep(self, result):
        """
        process result from statsmodels
        :param result: logistic regression result
        :return: dataframe with key statistics
        """
        # preparing outputs
        results_as_html = result.summary().tables[0].as_html()
        converged = pl.read_html(results_as_html)[0].iloc[5, 1]
        results_as_html = result.summary().tables[1].as_html()
        res = pd.read_html(results_as_html, header=0, index_col=0)[0]
        p_value = result.pvalues[self.indep_var_of_interest]
        beta_ind = result.params[self.indep_var_of_interest]
        conf_int_1 = res.loc[self.indep_var_of_interest]['[0.025']
        conf_int_2 = res.loc[self.indep_var_of_interest]['0.975]']

    def logistic_regression(self, phecode):
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
            cases = cases.with_columns(pl.Series([1] * len(cases)).alias("case"))
            controls = controls.with_columns(pl.Series([0] * len(controls)).alias("case"))

            # merge cases & controls
            regressors = pl.concat(cases, controls)

            # logistic regression
            regressors = sm.tools.add_constant(regressors[analysis_covariate_cols].to_numpy())
            logit = sm.Logit(regressors["case"].to_numpy(), regressors, missing='drop')
            result = logit.fit(disp=False)

            # choose to see results on the fly
            if self.verbose:
                print(result.summary())

    # now define function for running the phewas
    def run(self):
        """
        this method utilize multiprocessing tool and call runPheLogit() method.
        ---
        input
            no input needed, but required variables must be preloaded
            required variables must be loaded prior to running PheWAS:
                phecode_info
                ICD9_exclude
                sex_at_birth_restriction
        ---
        output
            dataframe containing phecode logistic regression results
        """

        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~    Running PheWAS   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

        # we will use multiprocessing tool for this
        manager = multiprocessing.Manager()
        self.return_dict = manager.dict()

        # this needs to have a particular list structure for the multiprocessing
        partitions = [
            list(ind) for ind in np.array_split(self.phecode_list, self.cores)
        ]
        pool = multiprocessing.Pool(processes=self.cores)

        # iterating over the list of indices 'partitions'
        # and subsetting the phenotypes to just the batch in index i of indices
        map_result = pool.map_async(self.logistic_regression, partitions)

        pool.close()
        pool.join()

        print("~~~~~~~~~~~~~~~~~~~~~~~~~    Processing Results    ~~~~~~~~~~~~~~~~~~~~~~~~~~~")

        # processing output
        logit_Phecode_results = [self.return_dict[k] for k in self.return_dict.keys()]
        logit_Phecode_results = pd.DataFrame(logit_Phecode_results)
        logit_Phecode_results.columns = ["phecode", "cases",
                                         "control", "p_value",
                                         "beta_ind", "conf_int_1",
                                         "conf_int_2", "converged"]
        logit_Phecode_results["code_val"] = logit_Phecode_results["phecode"].astype('float')
        logit_Phecode_results["neg_p_log_10"] = -np.log10(logit_Phecode_results["p_value"])
        logit_Phecode_results = pd.merge(phecode_info, logit_Phecode_results)

        # now save logit phecode as attribute
        self.logit_Phecode_results = logit_Phecode_results

        print("~~~~~~~~~~~~~~~~~~~~~~~~~~    PheWAS Completed    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
