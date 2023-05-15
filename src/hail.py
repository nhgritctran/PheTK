import hail as hl
import polars as pl
import pyarrow as pa


def _spark_to_polars(spark_df):

    polars_df = pl.from_arrow(pa.Table.from_batches(spark_df._collect_as_arrow()))

    return polars_df


def build_variant_cohort(mt_path,
                         chromosome_number,
                         genomic_position,
                         ref_allele,
                         alt_allele,
                         case_gt,
                         control_gt,
                         reference_genome="GRCh38",
                         db="aou"):
    # basic data processing
    gt_list = [case_gt, control_gt]
    alleles = f"{ref_allele}:{alt_allele}"
    base_locus = f"{chromosome_number}:{genomic_position}"
    if reference_genome == "GRCh38":
        locus = "chr" + base_locus
    elif reference_genome == "GRCh37":
        locus = base_locus
    else:
        return "Invalid reference version. Allowed inputs are \"GRCh37\" or \"GRCh38\"."
    variant = locus + alleles

    # initialize Hail
    if db == "aou":
        hl.init(default_reference=reference_genome)

    # hail variant struct
    variant = hl.parse_variant(variant, reference_genome=reference_genome)

    # load and filter matrix table
    mt = hl.read_matrix_table(mt_path)
    mt = mt.filter_rows(mt.locus == hl.Locus.parse(locus))
    if not mt:
        return f"Locus {locus} not found!"
    else:
        print(f"Locus {locus} found!")
        mt.row.show()
        print()

    # split if multi-allelic site
    allele_count = _spark_to_polars(mt.entries().select("info").to_spark())
    allele_count = len(allele_count["info.AF"][0])
    if allele_count > 1:
        mt = hl.split_multi(mt)
        print("Matrix table after multi-allelic split:")
        mt.row.show()
        print()

    # keep variant of interest
    if mt.alleles == variant["alleles"]:
        mt = mt.filter_rows((mt.locus == variant["locus"]) & \
                            (mt.alleles == variant["alleles"]))
        print(f"Variant {variant} found!")
        mt.row.show()
        print()
    else:
        return f"Variant {variant} not found!"

    spark_df = mt.entries().select("GT").to_spark()
    polars_df = pl.from_arrow(pa.Table.from_batches(spark_df._collect_as_arrow()))

    polars_df = polars_df.filter(pl.col("GT").is_in(gt_list))
    polars_df = polars_df.with_columns(pl.when(pl.col("GT") == case_gt)
                                       .then(1)
                                       .otherwise(0)
                                       .alias("case"))
    print(polars_df["case"].sum(), "cases:", len(polars_df.filter(pl.col("case") == 0)), "controls")

    return polars_df
