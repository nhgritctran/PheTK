# noinspection PyUnresolvedReferences,PyProtectedMember
from phetk import _queries, _paths, _utils
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import os
import pandas as pd
import polars as pl
import sys


class Cohort:

    def __init__(self,
                 platform: str = "aou",
                 aou_db_version: int = 8,
                 aou_omop_cdr: str | None = None,
                 gbq_dataset_id: str | None = None):
        """
        Initialize Cohort object for generating genotype-based cohorts and adding covariates.
        
        Sets up database connections and validates platform-specific parameters for either
        All of Us or custom database platforms. Configures database version and CDR paths
        based on specified platform.
        
        :param platform: Database platform, currently supports "aou" (All of Us) or "custom".
        :type platform: str
        :param aou_db_version: Version of All of Us database (6-8), e.g., 7 for CDR v7.
        :type aou_db_version: int
        :param aou_omop_cdr: CDR string value defining where to query OMOP data, uses workspace CDR if None.
        :type aou_omop_cdr: str | None
        :param gbq_dataset_id: Google BigQuery dataset ID for custom platforms.
        :type gbq_dataset_id: str | None
        """
        self.aou_max_version = 8

        if platform.lower() != "aou" and platform.lower() != "custom":
            print("Unsupported database. Currently supports \"aou\" (All of Us) or \"custom\".")
            sys.exit(1)
        if platform.lower() == "aou" and (aou_db_version not in range(6, self.aou_max_version+1)):
            print(f"Unsupported database. Current All of Us (AoU) CDR version is {self.aou_max_version}. "
                  f"aou_db_version takes an integer value from 6 to {self.aou_max_version}. "
                  f"For other AoU database versions, "
                  f"please provide the AoU CDR string using aou_omop_cdr parameter instead.")
            sys.exit(1)
        if platform.lower() == "custom" and gbq_dataset_id is None:
            print("gbq_dataset_id is required for non All of Us platforms.")
            sys.exit(1)
        self.platform = platform.lower()

        # generate attributes for AoU class instance
        if self.platform == "aou":
            self.db_version = aou_db_version
            if aou_omop_cdr is None:
                self.cdr = os.getenv("WORKSPACE_CDR")
            else:
                self.cdr = aou_omop_cdr
            self.user_project = os.getenv("GOOGLE_PROJECT")
        else:
            self.cdr = gbq_dataset_id
                     
        # attributes for add_covariate method
        self.date_of_birth = False
        self.current_age = False
        self.current_age_squared = False
        self.current_age_cubed = False
        self.year_of_birth = False
        self.age_at_last_ehr_event = False
        self.age_at_last_ehr_event_squared = False
        self.age_at_last_ehr_event_cubed = False
        self.sex_at_birth = False
        self.last_ehr_date = False
        self.ehr_length = False
        self.dx_code_occurrence_count = False
        self.dx_condition_count = False
        self.genetic_ancestry = False
        self.first_n_pcs = 0

    def by_genotype(self,
                    chromosome_number: int,
                    genomic_position: int,
                    ref_allele: str,
                    alt_allele: str,
                    gt_dict: dict[int, str | list[str]] | None = None,
                    reference_genome: str = "GRCh38",
                    mt_path: str | None = None,
                    output_file_path: str | None = None) -> None:
        """
        Generate cohort based on genotype of variant of interest.
        
        Extracts genotype data from Hail matrix table for specified genomic variant,
        filters participants by requested genotype groups, and creates cohort file
        with person IDs and genotype labels. Handles multi-allelic sites and validates
        variant existence in dataset.
        
        :param chromosome_number: Chromosome number for variant location.
        :type chromosome_number: int
        :param genomic_position: Genomic position of variant on chromosome.
        :type genomic_position: int
        :param ref_allele: Reference allele for variant.
        :type ref_allele: str
        :param alt_allele: Alternative allele for variant.
        :type alt_allele: str
        :param gt_dict: Genotype mapping dictionary, e.g., {0: "0/0", 1: ["0/1", "1/1"]}.
        :type gt_dict: dict[int, str | list[str]] | None
        :param reference_genome: Reference genome version, accepts "GRCh37" or "GRCh38".
        :type reference_genome: str
        :param mt_path: Path to population level Hail variant matrix table.
        :type mt_path: str | None
        :param output_file_path: Path of output TSV file.
        :type output_file_path: str | None
        :return: Creates genotype cohort TSV file with person IDs and genotype labels.
        :rtype: None
        """

        # import hail and assign hail_init attribute if needed
        import hail as hl

        # set the database path
        if self.platform == "aou":
            if (mt_path is None) and (self.db_version in range(6, self.aou_max_version+1)):
                mt_path = getattr(_paths, f"cdr{self.db_version}_mt_path")

        elif self.platform == "custom" and mt_path is None:
            print("For custom platform, mt_path must not be None.")
            sys.exit(1)

        # basic data processing
        if output_file_path is None:
            output_file_path = "aou_chr" + \
                               str(chromosome_number) + "_" + \
                               str(genomic_position) + "_" + \
                               str(ref_allele) + "_" + \
                               str(alt_allele) + \
                               ".tsv"
        
        # prepare genotype dict and check for duplicated genotypes
        gt_list = []
        for v in gt_dict.values():
            if isinstance(v, str):
                v = [v]
            gt_list.extend(v)
        if _utils.has_overlapping_values(gt_dict):
            print("Error: Duplicated genotype(s) detected in genotype dict.")
            sys.exit(1)

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
        try:
            hl.init(default_reference=reference_genome)
        except Exception as err:
            if "IllegalArgumentException" not in str(err):
                raise
            else:
                print("Hail Initialization skipped as Hail has already been initialized.")

        # hail variant struct
        variant = hl.parse_variant(variant_string, reference_genome=reference_genome)

        # load and filter matrix table
        mt = hl.read_matrix_table(mt_path)
        mt = mt.filter_rows(mt.locus == hl.Locus.parse(locus))
        if mt.count_rows() == 0:
            print()
            print(f"\033[1mLocus {locus} not found!\033[0m")
            return
        elif mt.count_rows() >= 1:
            print()
            print(f"\033[1mLocus {locus} found!\033[0m")
            mt.row.show()

        # split if multi-allelic site
        allele_count = _utils.spark_to_polars(mt.entries().select("info").to_spark())
        allele_count = len(allele_count["info.AF"][0])
        if allele_count > 1:
            print()
            print("\033[1mMulti-allelic detected! Splitting...\033[0m")
            mt = hl.split_multi_hts(mt)
            mt.row.show()

        # keep variant of interest
        mt = mt.filter_rows((mt.locus == variant["locus"]) &
                            (mt.alleles == variant["alleles"]))
        if mt.count_rows() >= 1:
            print()
            print(f"\033[1mVariant {variant_string} found!\033[0m")
            mt.row.show()

            # export to polars
            spark_df = mt.entries().select("GT").to_spark()
            polars_df = _utils.spark_to_polars(spark_df)

            # convert the list of int to GT string, e.g., "0/0", "0/1", "1/1"
            polars_df = polars_df.with_columns(
                pl.col("GT.alleles").list.get(0).cast(pl.Utf8).alias("GT0"),
                pl.col("GT.alleles").list.get(1).cast(pl.Utf8).alias("GT1"),
            )
            polars_df = polars_df.with_columns((pl.col("GT0") + "/" + pl.col("GT1")).alias("GT"))
            
            # keep only participants with genotypes of interest
            polars_df = polars_df.filter(pl.col("GT").is_in(gt_list))
            # map genotype to their int values
            lookup = {}
            for key, value in gt_dict.items():
                if isinstance(value, list):
                    for v in value:
                        lookup[v] = key
                else:
                    lookup[value] = key
            polars_df = polars_df.with_columns(
                pl.col("GT").map_elements(lambda x: lookup.get(x), return_dtype=pl.Int64).alias("genotype")
            )

            cohort = polars_df \
                .rename({"s": "person_id"})[["person_id", "genotype"]] \
                .with_columns(pl.col("person_id").cast(int))
            cohort = cohort.unique()
            cohort.write_csv(output_file_path, separator="\t")

            print()
            cohort_gt = cohort["genotype"].unique().to_list()
            print(f"Cohort size: {len(cohort)} participants")
            for gt in cohort_gt:
                print(f"Genotype {gt}: {len(cohort.filter(pl.col('genotype')==gt))} participants")
            print()
            print(f"Cohort data saved as \033[1m{output_file_path}\033[0m")
            print()

        else:
            print()
            print(f"Variant {variant_string} not found!")
            print()

    def _get_ancestry_preds(
            self,
            user_project: str,
            participant_ids: tuple[int, ...]
    ) -> pl.DataFrame | None:
        """
        Retrieve ancestry predictions and principal components for specified participants.
        
        Loads ancestry prediction data from All of Us CDR, extracts and formats
        principal components (PCs) from pca_features, and filters for requested
        participant IDs. Method specifically designed for All of Us database.
        
        :param user_project: Google Cloud project ID for requester pays access.
        :type user_project: str
        :param participant_ids: Participant IDs to retrieve ancestry data for.
        :type participant_ids: tuple[int, ...]
        :return: Ancestry predictions and PCs dataframe, or None if data unavailable.
        :rtype: pl.DataFrame | None
        """
        n_pc = 16  # AoU specific
        if self.db_version in range(6, self.aou_max_version+1):
            ancestry_preds_file_path = getattr(_paths, f"cdr{self.db_version}_ancestry_pred_path")
            ancestry_preds = pd.read_csv(ancestry_preds_file_path,
                                         sep="\t",
                                         storage_options={"requester_pays": True,
                                                          "user_project": user_project})
            ancestry_preds = pl.from_pandas(ancestry_preds)
            ancestry_preds = ancestry_preds.with_columns(pl.col("pca_features").str.replace(r"\[", "")) \
                .with_columns(pl.col("pca_features").str.replace(r"\]", "")) \
                .with_columns(pl.col("pca_features").str.split(",").list.get(i).alias(f"pc{i+1}") for i in range(n_pc)) \
                .with_columns(pl.col(f"pc{i}").str.replace(" ", "").cast(float) for i in range(1, n_pc+1)) \
                .drop(["probabilities", "pca_features", "ancestry_pred_other"]) \
                .rename({"research_id": "person_id",
                         "ancestry_pred": "genetic_ancestry"}) \
                .filter(pl.col("person_id").is_in(participant_ids))
        else:
            ancestry_preds = None

        return ancestry_preds

    def _get_covariates(
            self,
            participant_ids: tuple[int, ...]
    ) -> pl.DataFrame:
        """
        Generate covariate data for specified participant IDs.
        
        Core internal function that retrieves demographic, clinical, and genetic
        covariates based on instance configuration. Queries All of Us database
        for age, sex, EHR statistics, and ancestry data as requested.
        
        :param participant_ids: Participant IDs to retrieve covariates for.
        :type participant_ids: tuple[int, ...]
        :return: Dataframe containing requested covariates for participants.
        :rtype: pl.DataFrame
        """

        # initial data prep
        if isinstance(participant_ids, str) or isinstance(participant_ids, int):
            participant_ids = (participant_ids,)
        elif isinstance(participant_ids, list):
            participant_ids = tuple(participant_ids)
        df = pl.DataFrame({"person_id": participant_ids})
        participant_ids = tuple([int(i) for i in participant_ids])

        # GET COVARIATES

        # current_age
        if self.current_age or self.current_age_squared or self.current_age_cubed or self.date_of_birth or self.year_of_birth:
            current_age_df = _utils.polars_gbq(_queries.current_age_query(self.cdr, participant_ids))
            cols_to_keep = ["person_id"]
            if self.current_age:
                cols_to_keep.append("current_age")
            if self.current_age_squared:
                cols_to_keep.append("current_age_squared")
            if self.current_age_cubed:
                cols_to_keep.append("current_age_cubed")
            if self.date_of_birth:
                cols_to_keep.append("date_of_birth")
            if self.year_of_birth:
                cols_to_keep.append("year_of_birth")
            df = df.join(current_age_df[cols_to_keep], how="left", on="person_id")

        # age_at_last_ehr_event, ehr_length, dx_code_occurrence_count, dx_condition_count
        if (self.age_at_last_ehr_event or self.age_at_last_ehr_event_squared or self.age_at_last_ehr_event_cubed or self.ehr_length or self.dx_code_occurrence_count
                or self.dx_condition_count or self.last_ehr_date):
            temp_df = _utils.polars_gbq(_queries.ehr_dx_code_query(self.cdr, participant_ids))
            cols_to_keep = ["person_id"]
            if self.age_at_last_ehr_event:
                cols_to_keep.append("age_at_last_ehr_event")
            if self.age_at_last_ehr_event_squared:
                cols_to_keep.append("age_at_last_ehr_event_squared")
            if self.age_at_last_ehr_event_cubed:
                cols_to_keep.append("age_at_last_ehr_event_cubed")
            if self.last_ehr_date:
                cols_to_keep.append("last_ehr_date")
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
                cols_to_keep = cols_to_keep + [f"pc{i}" for i in range(1, self.first_n_pcs+1)]
            df = df.join(temp_df[cols_to_keep], how="left", on="person_id")

        return df

    def add_covariates(
            self,
            cohort_file_path: str | None = None,
            date_of_birth: bool = False,
            year_of_birth: bool = False,
            current_age: bool = False,
            current_age_squared: bool = False,
            current_age_cubed: bool = False,
            sex_at_birth: bool = True,
            last_ehr_date: bool = False,
            age_at_last_ehr_event: bool = False,
            age_at_last_ehr_event_squared: bool = False,
            age_at_last_ehr_event_cubed: bool = False,
            ehr_length: bool = False,
            dx_code_occurrence_count: bool = False,
            dx_condition_count: bool = False,
            genetic_ancestry: bool = False,
            first_n_pcs: int = 0,
            chunk_size: int = 10000,
            drop_nulls: bool = False,
            output_file_path: str | None = None
    ) -> None:
        """
        Add demographic, clinical, and genetic covariates to existing cohort.
        
        Retrieves specified covariates from database and merges with input cohort file.
        Uses multi-threading for efficient processing of large cohorts. Supports
        various demographic, EHR-derived, and genetic ancestry variables.
        
        :param cohort_file_path: Path to cohort CSV or TSV file containing person_id column.
        :type cohort_file_path: str | None
        :param date_of_birth: Include participant date of birth.
        :type date_of_birth: bool
        :param year_of_birth: Include year of birth for participants.
        :type year_of_birth: bool
        :param current_age: Include current age of participants.
        :type current_age: bool
        :param current_age_squared: Include current age squared for participants.
        :type current_age_squared: bool
        :param current_age_cubed: Include current age cubed for participants.
        :type current_age_cubed: bool
        :param sex_at_birth: Include sex at birth from survey and observation data.
        :type sex_at_birth: bool
        :param last_ehr_date: Include date of last diagnosis event in EHR.
        :type last_ehr_date: bool
        :param age_at_last_ehr_event: Include age at last diagnosis event in EHR.
        :type age_at_last_ehr_event: bool
        :param age_at_last_ehr_event_squared: Include age at last diagnosis event squared.
        :type age_at_last_ehr_event_squared: bool
        :param age_at_last_ehr_event_cubed: Include age at last diagnosis event cubed.
        :type age_at_last_ehr_event_cubed: bool
        :param ehr_length: Include number of years that EHR record spans.
        :type ehr_length: bool
        :param dx_code_occurrence_count: Include count of diagnosis code occurrences on unique dates throughout EHR history.
        :type dx_code_occurrence_count: bool
        :param dx_condition_count: Include count of unique diagnosis conditions throughout EHR history.
        :type dx_condition_count: bool
        :param genetic_ancestry: Include predicted ancestry based on sequencing data.
        :type genetic_ancestry: bool
        :param first_n_pcs: Number of first principal components to include (0 for none).
        :type first_n_pcs: int
        :param chunk_size: Number of participant IDs per processing thread.
        :type chunk_size: int
        :param drop_nulls: Whether to drop rows with null values.
        :type drop_nulls: bool
        :param output_file_path: Path to output TSV file, can include \".tsv\" extension.
        :type output_file_path: str | None
        :return: Creates enhanced cohort TSV file with requested covariates.
        :rtype: None
        """
        # assign attributes
        self.date_of_birth = date_of_birth
        self.current_age = current_age
        self.current_age_squared = current_age_squared
        self.current_age_cubed = current_age_cubed
        self.year_of_birth = year_of_birth
        self.age_at_last_ehr_event = age_at_last_ehr_event
        self.age_at_last_ehr_event_squared = age_at_last_ehr_event_squared
        self.age_at_last_ehr_event_cubed = age_at_last_ehr_event_cubed
        self.sex_at_birth = sex_at_birth
        self.last_ehr_date = last_ehr_date
        self.ehr_length = ehr_length
        self.dx_code_occurrence_count = dx_code_occurrence_count
        self.dx_condition_count = dx_condition_count
        self.genetic_ancestry = genetic_ancestry
        self.first_n_pcs = first_n_pcs

        # check for valid input
        if cohort_file_path is not None:
            sep = _utils.detect_delimiter(cohort_file_path)
            cohort = pl.read_csv(cohort_file_path, separator=sep)
            if "person_id" not in cohort.columns:
                print("Cohort must contains \"person_id\" column!")
                sys.exit(1)
        else:
            print("A cohort is required."
                  "Please provide a valid file path.")
            sys.exit(1)

        # get participant IDs from cohort
        participant_ids = cohort["person_id"].unique().to_list()

        # set up multi-threading to generate covariates
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

        covariates = covariates.unique()
        covariates.write_csv("covariates.tsv", separator="\t")

        # merge covariates to cohort
        final_cohort = cohort.join(covariates, how="left", on="person_id")
        if drop_nulls:
            final_cohort = final_cohort.drop_nulls()

        if output_file_path is None:
            output_file_path = "cohort.tsv"
        final_cohort.write_csv(f"{output_file_path}", separator="\t")

        print()
        print(f"Cohort size: {len(final_cohort)} participants")
        if "genotype" in final_cohort.columns:
            cohort_gt = final_cohort["genotype"].unique().to_list()
            for gt in cohort_gt:
                print(f"Genotype {gt}: {len(final_cohort.filter(pl.col('genotype')==gt))} participants")
        print()
        print(f"Cohort data saved as \033[1m{output_file_path}\033[0m")
        print()


def main_by_genotype():
    """Main entry point for by-genotype CLI command."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate cohort based on genotype of variant of interest"
    )
    
    # Required arguments
    parser.add_argument("--chromosome_number", "-c", type=int, required=True,
                        help="Chromosome number for variant location")
    parser.add_argument("--genomic_position", "-p", type=int, required=True,
                        help="Genomic position of variant on chromosome")
    parser.add_argument("--ref_allele", "-r", type=str, required=True,
                        help="Reference allele for variant")
    parser.add_argument("--alt_allele", "-a", type=str, required=True,
                        help="Alternative allele for variant")
    parser.add_argument("--gt_dict", "-g", type=str, required=True,
                        help='Genotype mapping as JSON string, e.g., \'{"0": "0/0", "1": ["0/1", "1/1"]}\'')
    
    # Optional arguments
    parser.add_argument("--platform", type=str, default="aou",
                        help="Database platform: 'aou' or 'custom' (default: aou)")
    parser.add_argument("--aou_db_version", type=int, default=8,
                        help="Version of All of Us database (6-8) (default: 8)")
    parser.add_argument("--aou_omop_cdr", type=str, default=None,
                        help="CDR string value for OMOP data")
    parser.add_argument("--gbq_dataset_id", type=str, default=None,
                        help="Google BigQuery dataset ID for custom platforms")
    parser.add_argument("--reference_genome", type=str, default="GRCh38",
                        help="Reference genome version: 'GRCh37' or 'GRCh38' (default: GRCh38)")
    parser.add_argument("--mt_path", type=str, default=None,
                        help="Path to population level Hail variant matrix table")
    parser.add_argument("--output_file_path", "-o", type=str, default=None,
                        help="Name of output TSV file")
    
    args = parser.parse_args()
    
    # Parse genotype dict from JSON string
    import json
    try:
        gt_dict = json.loads(args.gt_dict)
        # Convert string keys to integers
        gt_dict = {int(k): v for k, v in gt_dict.items()}
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error parsing genotype_dict: {e}")
        print("Example format: '{\"0\": \"0/0\", \"1\": [\"0/1\", \"1/1\"]}'")
        sys.exit(1)
    
    # Create cohort instance
    cohort = Cohort(
        platform=args.platform,
        aou_db_version=args.aou_db_version,
        aou_omop_cdr=args.aou_omop_cdr,
        gbq_dataset_id=args.gbq_dataset_id
    )
    
    # Run by_genotype
    cohort.by_genotype(
        chromosome_number=args.chromosome_number,
        genomic_position=args.genomic_position,
        ref_allele=args.ref_allele,
        alt_allele=args.alt_allele,
        gt_dict=gt_dict,
        reference_genome=args.reference_genome,
        mt_path=args.mt_path,
        output_file_path=args.output_file_path
    )


def main_add_covariates():
    """Main entry point for add-covariates CLI command."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Add demographic, clinical, and genetic covariates to existing cohort"
    )
    
    # Required arguments
    parser.add_argument("--cohort_file_path", "-c", type=str, required=True,
                        help="Path to cohort CSV or TSV file containing person_id column")
    
    # Optional platform arguments
    parser.add_argument("--platform", type=str, default="aou",
                        help="Database platform: 'aou' or 'custom' (default: aou)")
    parser.add_argument("--aou_db_version", type=int, default=8,
                        help="Version of All of Us database (6-8) (default: 8)")
    parser.add_argument("--aou_omop_cdr", type=str, default=None,
                        help="CDR string value for OMOP data")
    parser.add_argument("--gbq_dataset_id", type=str, default=None,
                        help="Google BigQuery dataset ID for custom platforms")
    
    # Covariate arguments
    parser.add_argument("--date_of_birth", type=_utils.str_to_bool, default=False,
                        help="Include participant date of birth")
    parser.add_argument("--year_of_birth", type=_utils.str_to_bool, default=False,
                        help="Include year of birth for participants")
    parser.add_argument("--current_age", type=_utils.str_to_bool, default=False,
                        help="Include current age of participants")
    parser.add_argument("--current_age_squared", type=_utils.str_to_bool, default=False,
                        help="Include current age squared for participants")
    parser.add_argument("--current_age_cubed", type=_utils.str_to_bool, default=False,
                        help="Include current age cubed for participants")
    parser.add_argument("--sex_at_birth", type=_utils.str_to_bool, default=True,
                        help="Include sex at birth from survey and observation data")
    parser.add_argument("--last_ehr_date", type=_utils.str_to_bool, default=False,
                        help="Include date of last diagnosis event in EHR")
    parser.add_argument("--age_at_last_ehr_event", type=_utils.str_to_bool, default=False,
                        help="Include age at last diagnosis event in EHR")
    parser.add_argument("--age_at_last_ehr_event_squared", type=_utils.str_to_bool, default=False,
                        help="Include age at last diagnosis event squared")
    parser.add_argument("--age_at_last_ehr_event_cubed", type=_utils.str_to_bool, default=False,
                        help="Include age at last diagnosis event cubed")
    parser.add_argument("--ehr_length", type=_utils.str_to_bool, default=False,
                        help="Include number of years that EHR record spans")
    parser.add_argument("--dx_code_occurrence_count", type=_utils.str_to_bool, default=False,
                        help="Include count of diagnosis code occurrences on unique dates")
    parser.add_argument("--dx_condition_count", type=_utils.str_to_bool, default=False,
                        help="Include count of unique diagnosis conditions")
    parser.add_argument("--genetic_ancestry", type=_utils.str_to_bool, default=False,
                        help="Include predicted ancestry based on sequencing data")
    parser.add_argument("--first_n_pcs", type=int, default=0,
                        help="Number of first principal components to include (0 for none)")
    
    # Processing arguments
    parser.add_argument("--chunk_size", type=int, default=10000,
                        help="Number of participant IDs per processing thread (default: 10000)")
    parser.add_argument("--drop_nulls", type=_utils.str_to_bool, default=False,
                        help="Whether to drop rows with null values")
    parser.add_argument("--output_file_path", "-o", type=str, default=None,
                        help="Name for output TSV file")
    
    args = parser.parse_args()
    
    # Create cohort instance
    cohort = Cohort(
        platform=args.platform,
        aou_db_version=args.aou_db_version,
        aou_omop_cdr=args.aou_omop_cdr,
        gbq_dataset_id=args.gbq_dataset_id
    )
    
    # Run add_covariates
    cohort.add_covariates(
        cohort_file_path=args.cohort_file_path,
        date_of_birth=args.date_of_birth,
        year_of_birth=args.year_of_birth,
        current_age=args.current_age,
        current_age_squared=args.current_age_squared,
        current_age_cubed=args.current_age_cubed,
        sex_at_birth=args.sex_at_birth,
        last_ehr_date=args.last_ehr_date,
        age_at_last_ehr_event=args.age_at_last_ehr_event,
        age_at_last_ehr_event_squared=args.age_at_last_ehr_event_squared,
        age_at_last_ehr_event_cubed=args.age_at_last_ehr_event_cubed,
        ehr_length=args.ehr_length,
        dx_code_occurrence_count=args.dx_code_occurrence_count,
        dx_condition_count=args.dx_condition_count,
        genetic_ancestry=args.genetic_ancestry,
        first_n_pcs=args.first_n_pcs,
        chunk_size=args.chunk_size,
        drop_nulls=args.drop_nulls,
        output_file_path=args.output_file_path
    )
