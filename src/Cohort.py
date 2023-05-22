from . import genotype, covariate
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
        """
        This method is a proxy for genotype.build_variant_cohort method
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
        if cohort_csv_path is not None:
            cohort = pl.read_csv(cohort_csv_path)
            if "person_id" not in cohort.columns:
                print("Cohort must contains \"person_id\" column!")
            sys.exit(0)
        elif cohort_csv_path is None and self.genotype_cohort is not None:
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
                                              dx_code_occurrence_count=dx_code_occurrence_count,
                                              dx_condition_count=dx_condition_count,
                                              genetic_ancestry=genetic_ancestry,
                                              first_n_pcs=first_n_pcs,
                                              db_version=self.db_version,
                                              chunk_size=chunk_size)
        self.final_cohort = cohort.join(covariates, how="left", on="person_id")
        if drop_nulls:
            self.final_cohort = self.final_cohort.drop_nulls()
        self.final_cohort.write_csv("cohort.csv")

        print()
        print("\033[1mCohort size:", len(self.final_cohort))
        print("\033[1mCases:", self.final_cohort["case"].sum())
        print("\033[1mControls:", len(self.final_cohort.filter(pl.col("case") == 0)), "\033[0m")
        print()
        print("\033[1mCohort data saved as \"cohort.csv\"!\033[0m")
        print()
