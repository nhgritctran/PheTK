from . import _paths, _utils, covariate
import hail as hl
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
                               str(chromosome_number) + \
                               str(genomic_position) + \
                               str(ref_allele) + \
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
            cohort.write_csv(output_file_name)

            print()
            print("\033[1mCohort size:", len(cohort))
            print("\033[1mCases:", cohort["case"].sum())
            print("\033[1mControls:", len(cohort.filter(pl.col("case") == 0)), "\033[0m")
            print()
            print(f"\033[1mCohort data saved as {output_file_name}!\033[0m")
            print()

            self.genotype_cohort = cohort

        else:
            print()
            print(f"Variant {variant_string} not found!")
            print()

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
