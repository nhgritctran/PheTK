from . import utils
import hail as hl
import polars as pl


def build_variant_cohort(mt_path,
                         chromosome_number,
                         genomic_position,
                         ref_allele,
                         alt_allele,
                         case_gt,
                         control_gt,
                         reference_genome="GRCh38",
                         db="aou",
                         output_file_name=None):
    """
    generate cohort based on genotype of variant of interest
    :param mt_path: path to population level Hail variant matrix table
    :param chromosome_number: chromosome number; int
    :param genomic_position: genomic position; int
    :param ref_allele: reference allele; str
    :param alt_allele: alternative allele; str
    :param case_gt: genotype(s) for case; str or list of str
    :param control_gt: genotype(s) for control; str or list of str
    :param reference_genome: defaults to "GRCh38"; accepts "GRCh37" or "GRCh38"
    :param db: defaults to "aou"; accepts "aou" or "ukb"
    :param output_file_name: name of csv file output
    :return: polars data
    """
    # basic data processing
    if output_file_name:
        output_file_name = f"{output_file_name}.csv"
    else:
        output_file_name = "cohort.csv"
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
    if db == "aou":
        hl.init(default_reference=reference_genome)

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
    allele_count = utils.spark_to_polars(mt.entries().select("info").to_spark())
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
        polars_df = utils.spark_to_polars(spark_df)

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
        cohort = polars_df.rename({"s": "person_id"})[["person_id", "case"]]
        print()
        print("\033[1mCohort size:", len(cohort))
        print("\033[1mCases:", cohort["case"].sum())
        print("\033[1mControls:", len(cohort.filter(pl.col("case") == 0)), "\033[0m")
        cohort.write_csv(output_file_name)
        print(f"\033[1mCohort data saved as {output_file_name}!\033[0m")
        return cohort

    else:
        print()
        print(f"Variant {variant_string} not found!")
        return
