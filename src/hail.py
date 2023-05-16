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
        return f"\033[1mLocus {locus} not found!"
    else:
        print()
        print(f"Locus {locus} found!")
        mt.row.show()

    # split if multi-allelic site
    allele_count = _spark_to_polars(mt.entries().select("info").to_spark())
    allele_count = len(allele_count["info.AF"][0])
    if allele_count > 1:
        print()
        print("Multi-allelic detected! Splitting...")
        mt = hl.split_multi(mt)
        mt.row.show()

    # keep variant of interest
    mt = mt.filter_rows((mt.locus == variant["locus"]) & \
                        (mt.alleles == variant["alleles"]))
    if mt:
        print()
        print(f"Variant {variant_string} found!")
        mt.row.show()

        # export to polars
        spark_df = mt.entries().select("GT").to_spark()
        polars_df = _spark_to_polars(spark_df)

        # convert list of int to GT string, e.g., "0/0", "0/1", "1/1"
        polars_df = polars_df.with_columns(
            pl.col("GT.alleles").arr.get(0).cast(pl.Utf8).alias("GT0"),
            pl.col("GT.alleles").arr.get(1).cast(pl.Utf8).alias("GT1"),
        )
        polars_df = polars_df.with_columns((pl.col("GT0") + "/" + pl.col("GT1")).alias("GT"))
        polars_df = polars_df.filter(pl.col("GT").is_in(gt_list))
        polars_df = polars_df.with_columns(pl.when(pl.col("GT") == case_gt)
                                           .then(1)
                                           .otherwise(0)
                                           .alias("case"))
        cohort = polars_df.rename({"s": "person_id"})[["person_id", "case"]]
        print()
        print("Cohort size:", len(cohort))
        print("Cases:", cohort["case"].sum())
        print("Controls:", len(cohort.filter(pl.col("case") == 0)), "\033[0m")
        print(cohort.head())

        return cohort

    else:
        print()
        return f"Variant {variant_string} not found!"
