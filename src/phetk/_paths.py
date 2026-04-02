import os

_TERRA_BUCKET = "fc-aou-datasets-controlled"
_VERILY_BUCKET = "vwb-aou-datasets-controlled"


def controlled_bucket() -> str:
    """Return the AoU controlled-tier bucket for the current workspace."""
    project = os.getenv("GOOGLE_PROJECT", "")
    if project.startswith("wb-"):
        return _VERILY_BUCKET
    return _TERRA_BUCKET


def cdr6_ancestry_pred_dir():
    return f"gs://{controlled_bucket()}/v6/wgs/vcf/aux/ancestry/"


def cdr7_ancestry_pred_dir():
    return f"gs://{controlled_bucket()}/v7/wgs/short_read/snpindel/aux/ancestry/"


def cdr8_ancestry_pred_dir():
    return f"gs://{controlled_bucket()}/v8/wgs/short_read/snpindel/aux/ancestry/"


def cdr6_mt_path():
    return f"gs://{controlled_bucket()}/v6/wgs/hail.mt"


def cdr7_mt_path():
    return f"gs://{controlled_bucket()}/v7/wgs/short_read/snpindel/acaf_threshold_v7.1/splitMT/hail.mt"


def cdr8_mt_path():
    return f"gs://{controlled_bucket()}/v8/wgs/short_read/snpindel/acaf_threshold/splitMT/hail.mt"

# obsolete paths
# cdr7_mt_path = "gs://fc-aou-datasets-controlled/v7/wgs/short_read/snpindel/acaf_threshold_v7.1/multiMT/hail.mt"
# cdr7_mt_path = "gs://fc-aou-datasets-controlled/v7/wgs/short_read/snpindel/acaf_threshold/multiMT/hail.mt"
