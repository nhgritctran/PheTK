# load libraries

# standard libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, date
from IPython.display import display, HTML

# PheWAS_Pool
import traceback
import multiprocessing
import statsmodels.api as sm


class PheWAS_Pool:
    """
    Class for performing PheWAS
    ---
    attributes:

        phecode_counts: Pandas Dataframe of Phecodes

        covariates: Pandas Dataframe of covariates to include in the analysis

        indep_var: String indicating the column in covariates that is the
                   independent variable of interest

        CDR_version: String indicating CDR version. Removed by HM

        phecode_process: list for phecodes to process

        min_cases: minimum number of cases for an individual phenotype to be analyzed

        cores: if not "", then specify number of cores to use in the analysis
               default updated by HM to multiprocessing.cpu_count() - 1
    ---
    methods:
        .run() method to run PheWAS analysis
        .logit_Phecode_results() method to retrieve analysis output
    ---
    REQUIRED variables must be loaded prior to running PheWAS:
        phecode_info
        phecode_counts
        ICD9_exclude
        sex_at_birth_restriction
    """

    def __init__(self,
                 phecode_counts,
                 covariates,
                 indep_var_of_interest="",
                 phecode_process='all',
                 min_cases=100,
                 genderspec_independent_var_names=["age_at_last_event",
                                                   "ehr_length",
                                                   "code_cnt"],
                 independent_var_names=["age_at_last_event",
                                        "ehr_length",
                                        "code_cnt",
                                        "male"],
                 show_res=False,
                 cores=multiprocessing.cpu_count() - 1):  # cores default updated by HM

        print("~~~~~~~~~~~~~~~~~~~~~    Creating PheWAS AOU Object    ~~~~~~~~~~~~~~~~~~~~~~~")

        # create instance attributes
        self.indep_var_of_interest = indep_var_of_interest
        # update 09_5_2019: only process phecodes passed in phecode counts
        if phecode_process == 'all':
            self.phecode_list = phecode_counts["phecode"].unique().tolist()
        else:
            self.phecode_list = phecode_process
        #         self.CDR_version = CDR_version
        self.cores = cores

        print("~~~~~~~~~~~~~~~~~~~    Merging Phecodes and Covariates    ~~~~~~~~~~~~~~~~~~~~")

        self.demo_patients_phecodes = pd.merge(covariates, phecode_counts, on="person_id")
        self.show_res = show_res
        self.independent_var_names = independent_var_names
        self.independent_var_names = list(np.append(np.array([self.indep_var_of_interest]),
                                                    self.independent_var_names))
        self.genderspec_independent_var_names = genderspec_independent_var_names
        self.genderspec_independent_var_names = list(np.append(np.array([self.indep_var_of_interest]),
                                                               self.genderspec_independent_var_names))
        self.remove_dup = list(np.append(np.array(["person_id"]),
                                         self.independent_var_names))
        self.min_cases = min_cases

    def runPheLogit(self, phecodes, min_count=2):
        """
        PheWAS logistic regression
        ---
        input
            phecodes: unqiue phecode list, partitioned from .run() method
            min_count: minimum phecode count to define cases, default = 2
        ---
        output
            self.return_dict: key = phecode
                              value = phecode,cases.shape[0],
                                      control.shape[0],
                                      p_value,
                                      beta_ind,
                                      conf_int_1,
                                      conf_int_2,
                                      converged
        """
        # diagnostics
        # min_count:
        for phecode in phecodes:
            # init error
            error = "Other Error"
            try:
                # First, need to define the exclusions, sufficient to just use the ICD9 one since the
                # overall phecodes are the same
                phecode_exclusions = ICD9_exclude[
                    ICD9_exclude["code"] == phecode
                    ][
                    "exclusion_criteria"
                ].unique().tolist()

                # we need to do sex specific counting here.
                # First find all people with at least 2 of the phecode
                cases = self.demo_patients_phecodes[
                    (self.demo_patients_phecodes["phecode"] == phecode) &
                    (self.demo_patients_phecodes["count"] >= min_count)]

                # now determine if there is a sex specific restriction
                # this is convoluted, but it is written this way to avoid storing
                # another copy of all the data in memory
                # HM updated phecode datatype as tring
                # sex_at_birth_restriction is a global variable
                male_only = sex_at_birth_restriction[
                                sex_at_birth_restriction["phecode"] == phecode
                                ]["male_only"] == True
                female_only = sex_at_birth_restriction[
                                  sex_at_birth_restriction["phecode"] == phecode
                                  ]["female_only"] == True

                # now, if sex at birth specific, filter all the people to the proper sex
                # and set the right covariates
                if len(np.array(male_only)) > 0 and np.array(male_only)[0] == True:
                    analysis_independent_var_names = self.genderspec_independent_var_names
                    cases = cases[cases['male'] == 1]  # restrict to males

                # if female only, then restrict to people who are female == 1
                elif len(np.array(female_only)) > 0 and np.array(female_only)[0] == True:
                    analysis_independent_var_names = self.genderspec_independent_var_names
                    cases = cases[cases['female'] == 1]  # restrict to females

                # otherwise there is no restriction, so we can use sex at birth in the analyses
                else:
                    analysis_independent_var_names = self.independent_var_names

                # Now cases have been properly modified and so we remove any duplicates and
                # restrict the analysis to just the regressors
                cases = cases[self.remove_dup].drop_duplicates()[analysis_independent_var_names]

                # Now test to see if we have enough cases
                # This is written like this to avoid unnecessary compute for phecodes
                # for which we don't have enough cases
                if cases.shape[0] >= self.min_cases:
                    # if it passes, create set all people that need to be excluded
                    exclude = self.demo_patients_phecodes[
                        (self.demo_patients_phecodes["phecode"].isin(
                            np.append(phecode_exclusions, phecode)))]

                    # if sex specific, restrict analyses to just that sex at birth
                    if len(np.array(male_only)) > 0 and np.array(male_only)[0] == True:
                        control = self.demo_patients_phecodes[
                            (self.demo_patients_phecodes.person_id.isin(exclude.person_id) == False) &
                            (self.demo_patients_phecodes.male == 1)]  # pick off just the males

                    # if female only, then restrict to people who are female == 1
                    elif len(np.array(female_only)) > 0 and np.array(female_only)[0] == True:
                        control = self.demo_patients_phecodes[
                            (self.demo_patients_phecodes.person_id.isin(exclude.person_id) == False) &
                            (self.demo_patients_phecodes.female == 1)]  # pick off just the males

                    # otherwise there is no restriction, so we can use sex at birth in the analyses
                    else:
                        control = self.demo_patients_phecodes[
                            self.demo_patients_phecodes.person_id.isin(exclude.person_id) == False]

                    # Now controls have been properly modified and so we remove
                    # any duplicates and restrict the analysis to just the regressors
                    control = control[self.remove_dup].drop_duplicates()[analysis_independent_var_names]

                    ###################################################################################
                    ## Perform Logistic regression
                    ## Now run through the logit function from stats models
                    ###################################################################################
                    y = [1] * cases.shape[0] + [0] * control.shape[0]
                    regressors = pd.concat([cases, control])
                    regressors = sm.tools.add_constant(regressors)
                    logit = sm.Logit(y, regressors, missing='drop')
                    result = logit.fit(disp=False)

                    # choose to see results on the fly
                    if self.show_res == True:
                        print(result.summary())
                    else:
                        pass

                    # preparing outputs
                    results_as_html = result.summary().tables[0].as_html()
                    converged = pd.read_html(results_as_html)[0].iloc[5, 1]
                    results_as_html = result.summary().tables[1].as_html()
                    res = pd.read_html(results_as_html, header=0, index_col=0)[0]
                    p_value = result.pvalues[self.indep_var_of_interest]
                    beta_ind = result.params[self.indep_var_of_interest]
                    conf_int_1 = res.loc[self.indep_var_of_interest]['[0.025']
                    conf_int_2 = res.loc[self.indep_var_of_interest]['0.975]']

                    # combining stat outputs and return
                    self.return_dict[phecode] = [phecode, cases.shape[0],
                                                 control.shape[0],
                                                 p_value,
                                                 beta_ind,
                                                 conf_int_1,
                                                 conf_int_2,
                                                 converged]

                else:
                    error = "Error in Phecode: " + str(phecode) + \
                            ": Number of cases less than minimum of " + str(self.min_cases)

                # add dummy 'control' and 'regressors' varibles in case they were not created
                # to avoid error 'control'/'regressors' referenced before assignment when del is called
                try:
                    control
                except NameError:
                    control = pd.DataFrame()
                    regressors = pd.DataFrame()
                del [control, cases, regressors]

            except Exception as e:
                print(error + ": " + str(e))
                # self.run_outcome_error.append(phecode)
                # pass

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
        map_result = pool.map_async(self.runPheLogit, partitions)

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