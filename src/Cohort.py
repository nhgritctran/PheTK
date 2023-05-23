from . import _paths, _queries, _utils
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.notebook import tqdm
import hail as hl
import os
import pandas as pd
import polars as pl
import sys


class Cohort:

    def __init__(self,
                 db="aou",
                 db_version=7):
        """
        :param db: database; currently supports "aou" (All of Us)
        :param db_version: int type, version of database, e.g., 7 for All of Us CDR v7
        """
        if db != "aou":
            print("Unsupported database. Currently supports \"aou\" (All of Us).")
            sys.exit(0)
        if db_version != 7:
            print("Unsupported database. Currently supports \"aou\" (All of Us) CDR v7 (enter 7 as parameter value).")
            sys.exit(0)
        self.db = db
        self.db_version = db_version
        self.cdr = os.getenv("WORKSPACE_CDR")
        self.user_project = os.getenv("GOOGLE_PROJECT")

        # attributes for add_covariate method
        self.natural_age = True
        self.age_at_last_event = True
        self.sex_at_birth = True
        self.ehr_length = True
        self.dx_code_occurrence_count = True
        self.dx_condition_count = True
        self.genetic_ancestry = False
        self.first_n_pcs = 0

        # output attributes
        self.genotype_cohort = None
        self.covariates = None
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
        """
        Generate cohort based on genotype of variant of interest
        :param chromosome_number: chromosome number; int
        :param genomic_position: genomic position; int
        :param ref_allele: reference allele; str
        :param alt_allele: alternative allele; str
        :param case_gt: genotype(s) for case; str or list of str
        :param control_gt: genotype(s) for control; str or list of str
        :param reference_genome: defaults to "GRCh38"; accepts "GRCh37" or "GRCh38"
        :param mt_path: path to population level Hail variant matrix table
        :param output_file_name: name of csv file output
        :return: genotype cohort csv file as well as polars dataframe object
        """
        # basic data processing
        if output_file_name:
            output_file_name = f"{output_file_name}.csv"
        else:
            output_file_name = "aou_chr" + \
                               str(chromosome_number) + "_" + \
                               str(genomic_position) + "_" + \
                               str(ref_allele) + "_" + \
                               str(alt_allele) + \
                               ".csv"
        if isinstance(case_gt, str):
            case_gt = [case_gt]
        if isinstance(control_gt, str):
            control_gt = [control_gt]
        gt_list = case_gt + control_gt
        alleles = f"{ref_allele}:{alt_allele}"
        base_locus = f"{chromosome_number}:{genomic_position}"
        if reference_genome == "GRCh38":
            locus = "chr" + base_locus
        elif reference_genome == "GRCh37":
            locus = base_locus
        else:
            print("Invalid reference version. Allowed inputs are \"GRCh37\" or \"GRCh38\".")
            return
        variant_string = locus + ":" + alleles

        # initialize Hail
        if self.db == "aou":
            hl.init(default_reference=reference_genome)
            if mt_path is None and self.db_version == 7:
                mt_path = _paths.cdr7_mt_path

        # hail variant struct
        variant = hl.parse_variant(variant_string, reference_genome=reference_genome)

        # load and filter matrix table
        mt = hl.read_matrix_table(mt_path)
        mt = mt.filter_rows(mt.locus == hl.Locus.parse(locus))
        if not mt:
            print()
            print(f"\033[1mLocus {locus} not found!")
            return
        else:
            print()
            print(f"\033[1mLocus {locus} found!")
            mt.row.show()

        # split if multi-allelic site
        allele_count = _utils.spark_to_polars(mt.entries().select("info").to_spark())
        allele_count = len(allele_count["info.AF"][0])
        if allele_count > 1:
            print()
            print("\033[1mMulti-allelic detected! Splitting...")
            mt = hl.split_multi(mt)
            mt.row.show()

        # keep variant of interest
        mt = mt.filter_rows((mt.locus == variant["locus"]) & \
                            (mt.alleles == variant["alleles"]))
        if mt:
            print()
            print(f"\033[1mVariant {variant_string} found!")
            mt.row.show()

            # export to polars
            spark_df = mt.entries().select("GT").to_spark()
            polars_df = _utils.spark_to_polars(spark_df)

            # convert list of int to GT string, e.g., "0/0", "0/1", "1/1"
            polars_df = polars_df.with_columns(
                pl.col("GT.alleles").arr.get(0).cast(pl.Utf8).alias("GT0"),
                pl.col("GT.alleles").arr.get(1).cast(pl.Utf8).alias("GT1"),
            )
            polars_df = polars_df.with_columns((pl.col("GT0") + "/" + pl.col("GT1")).alias("GT"))
            polars_df = polars_df.filter(pl.col("GT").is_in(gt_list))
            polars_df = polars_df.with_columns(pl.when(pl.col("GT").is_in(case_gt))
                                               .then(1)
                                               .otherwise(0)
                                               .alias("case"))
            cohort = polars_df \
                .rename({"s": "person_id"})[["person_id", "case"]] \
                .with_columns(pl.col("person_id").cast(int))
            cohort = cohort.unique()

            self.genotype_cohort = cohort
            self.genotype_cohort.write_csv(output_file_name)

            print()
            print("\033[1mCohort size:", len(cohort))
            print("\033[1mCases:", cohort["case"].sum())
            print("\033[1mControls:", len(cohort.filter(pl.col("case") == 0)), "\033[0m")
            print()
            print(f"\033[1mCohort data saved as {output_file_name}!\033[0m")
            print()

        else:
            print()
            print(f"Variant {variant_string} not found!")
            print()

    def _get_ancestry_preds(self, user_project, participant_ids):
        """
        This method specifically designed for All of Us database
        :param user_project: proxy of GOOGLE_PROJECT environment variable of current workspace in All of Us workbench
        :param participant_ids: participant IDs of interest
        :return: ancestry_preds data of specific version as polars dataframe object
        """
        if self.db_version == 7:
            ancestry_preds = pd.read_csv(_paths.cdr7_ancestry_pred_path,
                                         sep="\t",
                                         storage_options={"requester_pays": True,
                                                          "user_project": user_project})
            ancestry_preds = pl.from_pandas(ancestry_preds)
            ancestry_preds = ancestry_preds.with_columns(pl.col("pca_features").str.replace(r"\[", "")) \
                .with_columns(pl.col("pca_features").str.replace(r"\]", "")) \
                .with_columns(pl.col("pca_features").str.split(",").arr.get(i).alias(f"pc{i}") for i in range(16)) \
                .with_columns(pl.col(f"pc{i}").str.replace(" ", "").cast(float) for i in range(16)) \
                .drop(["probabilities", "pca_features", "ancestry_pred_other"]) \
                .rename({"research_id": "person_id",
                         "ancestry_pred": "genetic_ancestry"}) \
                .filter(pl.col("person_id").is_in(participant_ids))
        else:
            ancestry_preds = None

        return ancestry_preds

    def _get_covariates(self, participant_ids):
        """
        This method specifically designed for All of Us database
        Core internal function to generate covariate data for a set of participant IDs
        :param participant_ids: IDs of interest
        :return: polars dataframe object
        """

        # initial data prep
        if isinstance(participant_ids, str) or isinstance(participant_ids, int):
            participant_ids = (participant_ids,)
        elif isinstance(participant_ids, list):
            participant_ids = tuple(participant_ids)
        df = pl.DataFrame({"person_id": participant_ids})
        participant_ids = tuple([int(i) for i in participant_ids])

        # GET COVARIATES
        # natural_age
        if self.natural_age:
            natural_age_df = _utils.polars_gbq(_queries.natural_age_query(self.cdr, participant_ids))
            df = df.join(natural_age_df, how="left", on="person_id")

        # age_at_last_event, ehr_length, dx_code_occurrence_count, dx_condition_count
        if self.age_at_last_event or self.ehr_length or self.dx_code_occurrence_count or self.dx_condition_count:
            temp_df = _utils.polars_gbq(_queries.ehr_dx_code_query(self.cdr, participant_ids))
            cols_to_keep = ["person_id"]
            if self.age_at_last_event:
                cols_to_keep.append("age_at_last_event")
            if self.ehr_length:
                cols_to_keep.append("ehr_length")
            if self.dx_code_occurrence_count:
                cols_to_keep.append("dx_code_occurrence_count")
            if self.dx_condition_count:
                cols_to_keep.append("dx_condition_count")
            df = df.join(temp_df[cols_to_keep], how="left", on="person_id")

        # sex_at_birth
        if self.sex_at_birth:
            sex_df = _utils.polars_gbq(_queries.sex_at_birth(self.cdr, participant_ids))
            df = df.join(sex_df, how="left", on="person_id")

        # genetic_ancestry, first_n_pcs
        if self.genetic_ancestry or self.first_n_pcs > 0:
            temp_df = self._get_ancestry_preds(self.user_project, participant_ids)
            cols_to_keep = ["person_id"]
            if self.genetic_ancestry:
                cols_to_keep.append("genetic_ancestry")
            if self.first_n_pcs > 0:
                cols_to_keep = cols_to_keep + [f"pc{i}" for i in range(self.first_n_pcs)]
            df = df.join(temp_df[cols_to_keep], how="left", on="person_id")

        return df

    def add_covariates(self,
                       cohort_csv_path=None,
                       natural_age=True,
                       age_at_last_event=True,
                       sex_at_birth=True,
                       ehr_length=True,
                       dx_code_occurrence_count=True,
                       dx_condition_count=True,
                       genetic_ancestry=False,
                       first_n_pcs=0,
                       chunk_size=10000,
                       drop_nulls=False):
        """
        This method is a proxy for covariate.get_covariates method
        :param cohort_csv_path:
        :param natural_age: age of participants as of today
        :param age_at_last_event: age of participants at their last diagnosis event in EHR record
        :param sex_at_birth: sex at birth from survey and observation
        :param ehr_length: number of days that EHR record spans
        :param dx_code_occurrence_count: count of diagnosis code occurrences on unique dates,
                                         including ICD9CM, ICD10CM & SNOMED, throughout participant EHR history
        :param dx_condition_count: count of unique condition (dx code) throughout participant EHR history
        :param genetic_ancestry: predicted ancestry based on sequencing data
        :param first_n_pcs: number of first principal components to include
        :param chunk_size: defaults to 10,000; number of IDs per thread
        :param drop_nulls: defaults to False; drop rows having null values, i.e., participants without all covariates
        :return: csv file and polars dataframe object
        """
        # assign attributes
        self.natural_age = natural_age
        self.age_at_last_event = age_at_last_event
        self.sex_at_birth = sex_at_birth
        self.ehr_length = ehr_length
        self.dx_code_occurrence_count = dx_code_occurrence_count
        self.dx_condition_count = dx_condition_count
        self.genetic_ancestry = genetic_ancestry
        self.first_n_pcs = first_n_pcs

        # check for valid input
        if cohort_csv_path is not None:
            cohort = pl.read_csv(cohort_csv_path)
            if "person_id" not in cohort.columns:
                print("Cohort must contains \"person_id\" column!")
                sys.exit(0)
        elif cohort_csv_path is None and self.genotype_cohort is not None:
            cohort = self.genotype_cohort
        else:
            print("A cohort is required."
                  "Please run by_genotype() method to create a genotype cohort or provide a valid file path.")
            sys.exit(0)

        # get participant IDs from cohort
        participant_ids = cohort["person_id"].unique().to_list()

        # setup multi-threading to generate covariates
        chunks = [
            list(participant_ids)[i * chunk_size:(i + 1) * chunk_size] for i in
            range((len(participant_ids) // chunk_size) + 1)
        ]
        with ThreadPoolExecutor() as executor:
            jobs = [
                executor.submit(
                    self._get_covariates,
                    chunk
                ) for chunk in chunks
            ]
            result_list = [job.result() for job in tqdm(as_completed(jobs), total=len(chunks))]

        # process result
        result_list = [result for result in result_list if result is not None]
        covariates = result_list[0]
        for i in range(1, len(chunks)):
            covariates = pl.concat([covariates, result_list[i]])

        self.covariates = covariates.unique()
        self.covariates.write_csv("covariates.csv")

        # merge covariates to cohort
        final_cohort = cohort.join(covariates, how="left", on="person_id")
        if drop_nulls:
            final_cohort = final_cohort.drop_nulls()

        self.final_cohort = final_cohort
        self.final_cohort.write_csv("cohort.csv")

        print()
        print("\033[1mCohort size:", len(self.final_cohort))
        print("\033[1mCases:", self.final_cohort["case"].sum())
        print("\033[1mControls:", len(self.final_cohort.filter(pl.col("case") == 0)), "\033[0m")
        print()
        print("\033[1mCohort data saved as \"cohort.csv\"!\033[0m")
        print()
