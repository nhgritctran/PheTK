# noinspection PyUnresolvedReferences,PyProtectedMember
from phetk import _queries, _paths, _utils
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import os
import pandas as pd
import polars as pl
from google.cloud import storage
import sys
import warnings


class Cohort:

    def __init__(self,
                 platform: str = "aou",
                 aou_db_version: int = 8,
                 gbq_dataset_id: str | None = None):
        """
        Initialize Cohort object for generating genotype-based cohorts and adding covariates.

        Sets up database connections and validates platform-specific parameters for either
        All of Us or custom database platforms. Configures database version and CDR paths
        based on specified platform.

        Args:
            platform: Database platform, currently supports "aou" (All of Us) or "custom".
            aou_db_version: Version of All of Us database (6-8), e.g., 7 for CDR v7.
            gbq_dataset_id: BigQuery dataset ID. Overrides WORKSPACE_CDR on AoU. Required for custom platform.
        """
        self.aou_max_version = 8

        if platform.lower() != "aou" and platform.lower() != "custom":
            print("Unsupported database. Currently supports \"aou\" (All of Us) or \"custom\".")
            sys.exit(1)
        if platform.lower() == "aou" and (aou_db_version not in range(6, self.aou_max_version+1)):
            print(f"Unsupported database. Current All of Us (AoU) CDR version is {self.aou_max_version}. "
                  f"aou_db_version takes an integer value from 6 to {self.aou_max_version}. "
                  f"For other AoU database versions, "
                  f"please provide the CDR string using gbq_dataset_id parameter instead.")
            sys.exit(1)
        if platform.lower() == "custom" and gbq_dataset_id is None:
            print("gbq_dataset_id is required for non All of Us platforms.")
            sys.exit(1)
        self.platform = platform.lower()

        # generate attributes for AoU class instance
        if self.platform == "aou":
            self.db_version = aou_db_version
            if gbq_dataset_id is not None:
                self.cdr = gbq_dataset_id
            else:
                self.cdr = os.getenv("WORKSPACE_CDR")
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
                    data_format: str = "vcf",
                    call_set: str = "acaf_threshold",
                    data_path: str | None = None,
                    mt_path: str | None = None,
                    output_file_path: str | None = None) -> None:
        """
        Generate cohort based on genotype of variant of interest.

        Extracts genotype data from VCF or Hail matrix table for specified genomic
        variant, filters participants by requested genotype groups, and creates cohort
        file with person IDs and genotype labels. Handles multi-allelic sites.

        Args:
            chromosome_number: Chromosome number for variant location.
            genomic_position: Genomic position of variant on chromosome.
            ref_allele: Reference allele for variant.
            alt_allele: Alternative allele for variant.
            gt_dict: Genotype mapping dictionary, e.g., {0: "0/0", 1: ["0/1", "1/1"]}.
            reference_genome: Reference genome version, accepts "GRCh37" or "GRCh38".
            data_format: Genotype data format: "vcf" (default) or "hail".
            call_set: AoU callset name for path construction: "acaf_threshold" (default) or "exome".
            data_path: Override path to genotype data. For vcf, path to a VCF file
                (.vcf.gz, .vcf.bgz, .bcf) or AoU shard directory. For hail, the .mt directory path.
            mt_path: Deprecated. Use data_path instead. Kept for backward compatibility.
            output_file_path: Path of output TSV file.
        """
        # handle deprecated mt_path
        if mt_path is not None and data_path is not None:
            print("Error: Cannot specify both mt_path and data_path. Use data_path only.")
            sys.exit(1)
        if mt_path is not None:
            warnings.warn(
                "mt_path is deprecated. Use data_path instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            data_path = mt_path

        # validate data_format
        if data_format not in ("vcf", "hail"):
            print(f"Error: Invalid data_format \"{data_format}\". Must be \"vcf\" or \"hail\".")
            sys.exit(1)

        # shared input validation
        gt_list, gt_lookup, locus, variant_string, output_file_path = \
            self._validate_by_genotype_inputs(
                chromosome_number, genomic_position, ref_allele, alt_allele,
                gt_dict, reference_genome, output_file_path,
            )

        # resolve data path
        data_path = self._resolve_data_path(
            data_format, call_set, data_path, chromosome_number,
        )

        # extract genotypes using format-specific method
        if data_format == "hail":
            polars_df = self._extract_genotypes_hail(
                data_path, locus, variant_string, reference_genome,
                getattr(self, "user_project", None),
            )
        else:
            polars_df = self._extract_genotypes_vcf(
                data_path, chromosome_number, genomic_position,
                ref_allele, alt_allele, reference_genome,
                getattr(self, "user_project", None),
            )

        # filter, map, and write output
        if polars_df is not None:
            cohort = self._filter_and_map_genotypes(polars_df, gt_list, gt_lookup)
            self._write_cohort_output(cohort, output_file_path)
        else:
            print()
            print(f"Variant {variant_string} not found!")
            print()

    # ------------------------------------------------------------------
    # Shared helpers for by_genotype
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_by_genotype_inputs(
        chromosome_number: int,
        genomic_position: int,
        ref_allele: str,
        alt_allele: str,
        gt_dict: dict[int, str | list[str]],
        reference_genome: str,
        output_file_path: str | None,
    ) -> tuple[list[str], dict[str, int], str, str, str]:
        """Validate inputs and return (gt_list, gt_lookup, locus, variant_string, output_file_path)."""
        if output_file_path is None:
            output_file_path = (
                f"aou_chr{chromosome_number}_{genomic_position}"
                f"_{ref_allele}_{alt_allele}.tsv"
            )

        # flatten gt_dict values into gt_list
        gt_list = []
        for v in gt_dict.values():
            if isinstance(v, str):
                v = [v]
            gt_list.extend(v)

        if _utils.has_overlapping_values(gt_dict):
            print("Error: Duplicated genotype(s) detected in genotype dict.")
            sys.exit(1)

        # build reverse lookup: GT string -> integer label
        gt_lookup: dict[str, int] = {}
        for key, value in gt_dict.items():
            if isinstance(value, list):
                for v in value:
                    gt_lookup[v] = key
            else:
                gt_lookup[value] = key

        # build locus and variant strings
        alleles = f"{ref_allele}:{alt_allele}"
        base_locus = f"{chromosome_number}:{genomic_position}"
        if reference_genome == "GRCh38":
            locus = "chr" + base_locus
        elif reference_genome == "GRCh37":
            locus = base_locus
        else:
            print("Invalid reference version. Allowed inputs are \"GRCh37\" or \"GRCh38\".")
            sys.exit(1)
        variant_string = locus + ":" + alleles

        return gt_list, gt_lookup, locus, variant_string, output_file_path

    def _resolve_data_path(
        self,
        data_format: str,
        call_set: str,
        data_path: str | None,
        chromosome_number: int,
    ) -> str:
        """Resolve the genotype data path based on platform and format."""
        if data_path is not None:
            return data_path

        if self.platform == "aou":
            if data_format == "vcf":
                env_path = os.getenv("WGS_ACAF_THRESHOLD_VCF_PATH")
                if env_path is not None:
                    return env_path
                return (
                    f"gs://{_paths.controlled_bucket()}/v{self.db_version}"
                    f"/wgs/short_read/snpindel/{call_set}/vcf/"
                )
            else:  # hail
                env_path = os.getenv("WGS_ACAF_THRESHOLD_SPLIT_HAIL_PATH")
                if env_path is not None:
                    return env_path
                mt_path_func = getattr(_paths, f"cdr{self.db_version}_mt_path", None)
                if mt_path_func is not None:
                    return mt_path_func()
                print("Hail split matrix table path is not available. "
                      "Please provide data_path directly.")
                sys.exit(1)
        else:
            # custom platform requires explicit data_path
            print(f"For custom platform, data_path is required.")
            sys.exit(1)

    @staticmethod
    def _filter_and_map_genotypes(
        polars_df: pl.DataFrame,
        gt_list: list[str],
        gt_lookup: dict[str, int],
    ) -> pl.DataFrame:
        """Filter GT column to gt_list, map to integer labels, return cohort DataFrame."""
        polars_df = polars_df.filter(pl.col("GT").is_in(gt_list))
        polars_df = polars_df.with_columns(
            pl.col("GT").map_elements(
                lambda x: gt_lookup.get(x), return_dtype=pl.Int64
            ).alias("genotype")
        )
        cohort = polars_df[["person_id", "genotype"]].unique()
        return cohort

    @staticmethod
    def _write_cohort_output(cohort: pl.DataFrame, output_file_path: str) -> None:
        """Write cohort TSV and print summary."""
        cohort.write_csv(output_file_path, separator="\t")
        print()
        cohort_gt = cohort["genotype"].unique().to_list()
        print(f"Cohort size: {len(cohort)} participants")
        for gt in cohort_gt:
            print(f"Genotype {gt}: {len(cohort.filter(pl.col('genotype') == gt))} participants")
        print()
        print(f"Cohort data saved as \033[1m{output_file_path}\033[0m")
        print()

    # ------------------------------------------------------------------
    # Format-specific genotype extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_genotypes_hail(
        data_path: str,
        locus: str,
        variant_string: str,
        reference_genome: str,
        user_project: str | None = None,
    ) -> pl.DataFrame | None:
        """Extract genotypes from a Hail split matrix table.

        Returns DataFrame with columns ["person_id", "GT"] or None if variant not found.
        """
        try:
            import hail as hl
        except ImportError:
            raise ImportError(
                "hail is required for data_format='hail'. "
                "Install it with: pip install phetk[hail]"
            )

        # initialize Hail
        hl.init(idempotent=True, gcs_requester_pays_configuration=user_project)
        hl.default_reference(reference_genome)

        # hail variant struct
        variant = hl.parse_variant(variant_string, reference_genome=reference_genome)

        # load and filter matrix table
        mt = hl.read_matrix_table(data_path)
        mt = mt.filter_rows(mt.locus == hl.Locus.parse(locus))
        if mt.count_rows() == 0:
            print()
            print(f"\033[1mLocus {locus} not found!\033[0m")
            return None
        elif mt.count_rows() >= 1:
            print()
            print(f"\033[1mLocus {locus} found!\033[0m")
            mt.row.show()

        # split if multi-allelic site
        allele_count = _utils.spark_to_polars(mt.key_cols_by().entries().select("info").to_spark())
        allele_count = len(allele_count["info.AF"][0])
        if allele_count > 1:
            print()
            print("\033[1mMulti-allelic detected! Splitting...\033[0m")
            mt = hl.split_multi_hts(mt)
            mt.row.show()

        # keep variant of interest
        mt = mt.filter_rows((mt.locus == variant["locus"]) &
                            (mt.alleles == variant["alleles"]))
        if mt.count_rows() == 0:
            return None

        print()
        print(f"\033[1mVariant {variant_string} found!\033[0m")
        mt.row.show()

        # export to polars
        spark_df = mt.key_cols_by().entries().select("s", "GT").to_spark()
        polars_df = _utils.spark_to_polars(spark_df)

        # convert the list of int to GT string, e.g., "0/0", "0/1", "1/1"
        polars_df = polars_df.with_columns(
            pl.col("GT.alleles").list.get(0).cast(pl.Utf8).alias("GT0"),
            pl.col("GT.alleles").list.get(1).cast(pl.Utf8).alias("GT1"),
        )
        polars_df = polars_df.with_columns(
            (pl.col("GT0") + "/" + pl.col("GT1")).alias("GT")
        )

        # rename sample ID column and cast to int
        polars_df = polars_df.rename({"s": "person_id"})
        polars_df = polars_df.with_columns(pl.col("person_id").cast(int))
        polars_df = polars_df[["person_id", "GT"]]
        return polars_df

    @staticmethod
    def _setup_gcs_for_pysam(user_project: str | None) -> None:
        """Configure environment variables for htslib GCS access.

        htslib (used by pysam) needs two env vars for GCS requester-pays buckets:
        - GCS_OAUTH_TOKEN: OAuth2 access token for authentication
        - GCS_REQUESTER_PAYS_PROJECT: billing project for requester-pays

        The GCS Python client uses ADC automatically, but htslib does not —
        the token must be provided explicitly via GCS_OAUTH_TOKEN.

        See: https://github.com/samtools/htslib/blob/develop/hfile_gcs.c
        """
        if not user_project:
            return

        os.environ["GCS_REQUESTER_PAYS_PROJECT"] = user_project

        # htslib needs an explicit OAuth token — get one from ADC
        if "GCS_OAUTH_TOKEN" not in os.environ:
            try:
                import google.auth
                import google.auth.transport.requests
                credentials, _ = google.auth.default()
                credentials.refresh(google.auth.transport.requests.Request())
                os.environ["GCS_OAUTH_TOKEN"] = credentials.token
            except Exception:
                pass  # fall back to metadata server on GCE

    @staticmethod
    def _chrom_sort_key(chrom_str: str) -> int:
        """Convert chromosome name to a sortable integer for genomic ordering."""
        c = chrom_str.replace("chr", "")
        if c == "X":
            return 23
        if c == "Y":
            return 24
        if c in ("M", "MT"):
            return 25
        try:
            return int(c)
        except ValueError:
            return 99

    @staticmethod
    def _find_vcf_shard(
        vcf_dir: str,
        chromosome_number: int,
        genomic_position: int,
        reference_genome: str,
        user_project: str | None = None,
    ) -> str:
        """Find the VCF shard file covering a given genomic position.

        Uses binary search on the sorted shard list. At each step, opens
        the midpoint shard's tabix index to read its first record position,
        then narrows the search. Requires ~log2(N) index reads instead of
        downloading all N interval lists.

        Args:
            vcf_dir: GCS directory path containing VCF shards.
            chromosome_number: Chromosome number.
            genomic_position: Genomic position on chromosome.
            reference_genome: "GRCh38" or "GRCh37".
            user_project: GCP project ID for requester-pays billing.

        Returns:
            Full path to the matching .vcf.bgz file.
        """
        import math
        import pysam
        from google.cloud import storage as gcs_storage

        if reference_genome == "GRCh38":
            contig = f"chr{chromosome_number}"
        else:
            contig = str(chromosome_number)

        client = gcs_storage.Client(project=user_project)

        # parse GCS URI into bucket and prefix
        path_no_scheme = vcf_dir[5:]  # strip "gs://"
        bucket_name, prefix = path_no_scheme.split("/", 1)
        if not prefix.endswith("/"):
            prefix += "/"

        # list and sort .vcf.bgz blobs by name (numbered shards → genomic order)
        print("Listing VCF shards...")
        bucket = client.bucket(bucket_name, user_project=user_project)
        blobs = list(client.list_blobs(bucket, prefix=prefix))
        vcf_blobs = sorted(
            [b for b in blobs if b.name.endswith(".vcf.bgz")
             and not b.name.endswith(".tbi")],
            key=lambda b: b.name,
        )

        if not vcf_blobs:
            print(f"Error: No .vcf.bgz files found in {vcf_dir}")
            sys.exit(1)

        n = len(vcf_blobs)
        max_steps = math.ceil(math.log2(n)) + 1 if n > 1 else 1
        target_key = (Cohort._chrom_sort_key(contig), genomic_position)

        print(f"Found {n} VCF shard(s). "
              f"Binary searching for {contig}:{genomic_position}...")
        pbar = tqdm(total=max_steps, desc="Binary search", unit="step")

        def _vcf_path(blob):
            return f"gs://{bucket_name}/{blob.name}"

        downloaded_tbi = []

        def _get_shard_start(vcf_path):
            """Read the tabix index and return (chrom_order, position) of the
            first record in this shard, or None if the shard is empty."""
            # track the .tbi file htslib downloads locally
            tbi_local = os.path.basename(vcf_path) + ".tbi"
            tbx = pysam.TabixFile(vcf_path)
            if os.path.exists(tbi_local):
                downloaded_tbi.append(tbi_local)
            try:
                for contig_name in tbx.contigs:
                    try:
                        first_line = next(iter(tbx.fetch(contig_name)))
                        pos = int(first_line.split("\t")[1])
                        return (Cohort._chrom_sort_key(contig_name), pos)
                    except StopIteration:
                        continue
            finally:
                tbx.close()
            return None

        # binary search: find the last shard whose start <= target
        lo, hi = 0, n - 1
        while lo < hi:
            mid = lo + (hi - lo + 1) // 2  # upper-mid to avoid infinite loop
            pbar.update(1)
            shard_start = _get_shard_start(_vcf_path(vcf_blobs[mid]))
            if shard_start is not None and shard_start <= target_key:
                lo = mid
            else:
                hi = mid - 1

        pbar.update(pbar.total - pbar.n)  # fill remaining steps
        pbar.close()
        result = _vcf_path(vcf_blobs[lo])
        print(f"Shard found: {result}")

        # clean up locally downloaded .tbi files
        for tbi_file in downloaded_tbi:
            try:
                os.remove(tbi_file)
            except OSError:
                pass

        return result

    @staticmethod
    def _extract_genotypes_vcf(
        data_path: str,
        chromosome_number: int,
        genomic_position: int,
        ref_allele: str,
        alt_allele: str,
        reference_genome: str,
        user_project: str | None = None,
    ) -> pl.DataFrame | None:
        """Extract genotypes from a VCF file using pysam.

        Returns DataFrame with columns ["person_id", "GT"] or None if variant not found.
        The data_path can be a direct VCF file path (.vcf.gz, .vcf.bgz, .bcf)
        or an AoU shard directory (shard lookup via tabix).
        Supports GCS requester-pays paths (gs://...) via htslib.
        """
        import pysam

        # configure htslib for GCS requester-pays access
        if data_path.startswith("gs://"):
            Cohort._setup_gcs_for_pysam(user_project)

        # determine VCF file path
        vcf_extensions = (".vcf.gz", ".vcf.bgz", ".bcf")
        if any(data_path.endswith(ext) for ext in vcf_extensions):
            vcf_file_path = data_path
        else:
            # AoU shard directory — find the right shard via tabix
            vcf_file_path = Cohort._find_vcf_shard(
                data_path, chromosome_number, genomic_position,
                reference_genome, user_project,
            )

        # build contig string
        if reference_genome == "GRCh38":
            contig = f"chr{chromosome_number}"
        else:
            contig = str(chromosome_number)

        locus_str = f"{contig}:{genomic_position}"

        # open VCF and fetch region
        print()
        print(f"Opening VCF file: {vcf_file_path}")
        vcf = pysam.VariantFile(vcf_file_path)

        # find the exact variant at this position
        print(f"Fetching region {locus_str}...")
        target_record = None
        try:
            for record in vcf.fetch(contig, genomic_position - 1, genomic_position):
                if record.pos != genomic_position:
                    continue
                if record.ref != ref_allele:
                    continue
                if record.alts and alt_allele in record.alts:
                    target_record = record
                    break
        except ValueError:
            print()
            print(f"\033[1mContig {contig} not found in VCF header!\033[0m")
            vcf.close()
            return None

        if target_record is None:
            print()
            print(f"\033[1mVariant {locus_str}:{ref_allele}:{alt_allele} not found!\033[0m")
            vcf.close()
            return None

        print()
        print(f"\033[1mVariant {locus_str}:{ref_allele}:{alt_allele} found!\033[0m")

        is_multiallelic = len(target_record.alts) > 1
        if is_multiallelic:
            print("\033[1mMulti-allelic detected!\033[0m")
            target_alt_idx = list(target_record.alts).index(alt_allele)
            target_allele_code = target_alt_idx + 1  # 0 = REF

        # extract genotypes for all samples
        print("Extracting genotypes...")
        valid_person_ids = []
        gt_strings = []
        for sample_name, sample in target_record.samples.items():
            gt = sample["GT"]
            if gt is None or None in gt:
                continue

            if not is_multiallelic:
                gt_str = f"{gt[0]}/{gt[1]}"
            else:
                # remap alleles to match Hail split_multi_hts behaviour:
                # REF(0)->0, target_alt->1, any other alt->0 (treated as ref)
                r1 = 1 if gt[0] == target_allele_code else 0
                r2 = 1 if gt[1] == target_allele_code else 0
                # Hail normalizes unphased calls to canonical order (lower first);
                # phased calls preserve order since haplotype assignment matters.
                if not sample.phased:
                    lo, hi = min(r1, r2), max(r1, r2)
                    gt_str = f"{lo}/{hi}"
                else:
                    gt_str = f"{r1}/{r2}"

            valid_person_ids.append(sample_name)
            gt_strings.append(gt_str)

        vcf.close()

        polars_df = pl.DataFrame({
            "person_id": valid_person_ids,
            "GT": gt_strings,
        }).with_columns(pl.col("person_id").cast(int))

        return polars_df

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

        Args:
            user_project: Google Cloud project ID for requester pays access.
            participant_ids: Participant IDs to retrieve ancestry data for.

        Returns:
            Ancestry predictions and PCs dataframe, or None if data unavailable.
        """
        n_pc = 16  # AoU specific
        if self.db_version in range(6, self.aou_max_version+1):
            ancestry_dir = getattr(_paths, f"cdr{self.db_version}_ancestry_pred_dir")()
            bucket_name = _paths.controlled_bucket()
            prefix = ancestry_dir.replace(f"gs://{bucket_name}/", "")

            client = storage.Client(project=user_project)
            bucket = client.bucket(bucket_name, user_project=user_project)
            ancestry_preds_file_path = None
            for blob in client.list_blobs(bucket, prefix=prefix):
                if blob.name.endswith("ancestry_preds.tsv"):
                    ancestry_preds_file_path = f"gs://{bucket_name}/{blob.name}"
                    break

            if ancestry_preds_file_path is None:
                print(f"No *ancestry_preds.tsv file found in {ancestry_dir}")
                return None

            ancestry_preds = pd.read_csv(ancestry_preds_file_path,
                                         sep="\t",
                                         storage_options={"requester_pays": True,
                                                          "user_project": user_project})
            ancestry_preds = pl.from_pandas(ancestry_preds)
            ancestry_preds = ancestry_preds.with_columns(pl.col("pca_features").str.replace(r"\[", "")) \
                .with_columns(pl.col("pca_features").str.replace(r"\]", "")) \
                .with_columns(pl.col("pca_features").str.split(",").list.get(i).alias(f"pc{i+1}") for i in range(n_pc)) \
                .with_columns(pl.col(f"pc{i}").str.replace(" ", "").cast(float) for i in range(1, n_pc+1)) \
                .drop(["probabilities", "pca_features", "ancestry_pred"]) \
                .rename({"research_id": "person_id",
                         "ancestry_pred_other": "genetic_ancestry"}) \
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

        Args:
            participant_ids: Participant IDs to retrieve covariates for.

        Returns:
            Dataframe containing requested covariates for participants.
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

        Args:
            cohort_file_path: Path to cohort CSV or TSV file containing person_id column.
            date_of_birth: Include participant date of birth.
            year_of_birth: Include year of birth for participants.
            current_age: Include current age of participants.
            current_age_squared: Include current age squared for participants.
            current_age_cubed: Include current age cubed for participants.
            sex_at_birth: Include sex at birth from survey and observation data.
            last_ehr_date: Include date of last diagnosis event in EHR.
            age_at_last_ehr_event: Include age at last diagnosis event in EHR.
            age_at_last_ehr_event_squared: Include age at last diagnosis event squared.
            age_at_last_ehr_event_cubed: Include age at last diagnosis event cubed.
            ehr_length: Include number of years that EHR record spans.
            dx_code_occurrence_count: Include count of diagnosis code occurrences on unique dates throughout EHR history.
            dx_condition_count: Include count of unique diagnosis conditions throughout EHR history.
            genetic_ancestry: Include predicted ancestry based on sequencing data.
            first_n_pcs: Number of first principal components to include (0 for none).
            chunk_size: Number of participant IDs per processing thread.
            drop_nulls: Whether to drop rows with null values.
            output_file_path: Path to output TSV file, can include \".tsv\" extension.
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
    parser.add_argument("--gbq_dataset_id", type=str, default=None,
                        help="BigQuery dataset ID. Overrides WORKSPACE_CDR on AoU. Required for custom platform.")
    parser.add_argument("--reference_genome", type=str, default="GRCh38",
                        help="Reference genome version: 'GRCh37' or 'GRCh38' (default: GRCh38)")
    parser.add_argument("--data_format", type=str, default="vcf",
                        choices=["hail", "vcf"],
                        help="Genotype data format (default: vcf)")
    parser.add_argument("--call_set", type=str, default="acaf_threshold",
                        help="AoU callset name for path construction (default: acaf_threshold)")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Override path to genotype data")
    parser.add_argument("--mt_path", type=str, default=None,
                        help="(Deprecated) Path to Hail matrix table. Use --data_path instead.")
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
        data_format=args.data_format,
        call_set=args.call_set,
        data_path=args.data_path,
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
    parser.add_argument("--gbq_dataset_id", type=str, default=None,
                        help="BigQuery dataset ID. Overrides WORKSPACE_CDR on AoU. Required for custom platform.")

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
