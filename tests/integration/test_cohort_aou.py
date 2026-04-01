"""
Integration tests for cohort.py — require AoU Workbench environment.
Skipped automatically outside AoU (WORKSPACE_CDR env var not set).
Run inside AoU with: pytest -m aou
"""
import os
import pytest
import polars as pl

from phetk.cohort import Cohort

aou = pytest.mark.skipif(
    not os.getenv("WORKSPACE_CDR"),
    reason="Requires AoU Workbench environment (WORKSPACE_CDR not set)"
)

# Small set of known participant IDs to limit query cost
TEST_PARTICIPANT_IDS = [100, 101, 102, 103, 104]


@pytest.fixture(scope="module")
def aou_cohort():
    return Cohort(platform="aou", aou_db_version=8)


@pytest.fixture(scope="module")
def small_cohort_file(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("cohort_aou")
    path = tmp / "small_cohort.tsv"
    df = pl.DataFrame({"person_id": TEST_PARTICIPANT_IDS})
    df.write_csv(str(path), separator="\t")
    return str(path)


@pytest.mark.aou
class TestCohortInitAou:
    def test_cdr_set_from_workspace(self, aou_cohort):
        assert aou_cohort.cdr is not None
        assert len(aou_cohort.cdr) > 0

    def test_user_project_set(self, aou_cohort):
        assert aou_cohort.user_project is not None


@pytest.mark.aou
class TestAddCovariatesAou:
    def test_add_current_age(self, aou_cohort, small_cohort_file, tmp_path):
        out = str(tmp_path / "cohort_with_age.tsv")
        aou_cohort.add_covariates(
            cohort_file_path=small_cohort_file,
            current_age=True,
            sex_at_birth=False,
            output_file_path=out,
        )
        result = pl.read_csv(out, separator="\t")
        assert "current_age" in result.columns
        assert len(result) == len(TEST_PARTICIPANT_IDS)

    def test_add_sex_at_birth(self, aou_cohort, small_cohort_file, tmp_path):
        out = str(tmp_path / "cohort_with_sex.tsv")
        aou_cohort.add_covariates(
            cohort_file_path=small_cohort_file,
            sex_at_birth=True,
            output_file_path=out,
        )
        result = pl.read_csv(out, separator="\t")
        assert "sex_at_birth" in result.columns
        sex_vals = result["sex_at_birth"].drop_nulls().to_list()
        assert all(v in (0, 1) for v in sex_vals)

    def test_add_ehr_covariates(self, aou_cohort, small_cohort_file, tmp_path):
        out = str(tmp_path / "cohort_with_ehr.tsv")
        aou_cohort.add_covariates(
            cohort_file_path=small_cohort_file,
            ehr_length=True,
            dx_condition_count=True,
            sex_at_birth=False,
            output_file_path=out,
        )
        result = pl.read_csv(out, separator="\t")
        assert "ehr_length" in result.columns
        assert "dx_condition_count" in result.columns

    def test_add_ancestry_pcs(self, aou_cohort, small_cohort_file, tmp_path):
        out = str(tmp_path / "cohort_with_pcs.tsv")
        aou_cohort.add_covariates(
            cohort_file_path=small_cohort_file,
            genetic_ancestry=True,
            first_n_pcs=5,
            sex_at_birth=False,
            output_file_path=out,
        )
        result = pl.read_csv(out, separator="\t")
        assert "genetic_ancestry" in result.columns
        for i in range(1, 6):
            assert f"pc{i}" in result.columns

    def test_output_preserves_all_participants(self, aou_cohort, small_cohort_file, tmp_path):
        out = str(tmp_path / "cohort_all.tsv")
        aou_cohort.add_covariates(
            cohort_file_path=small_cohort_file,
            current_age=True,
            sex_at_birth=False,
            output_file_path=out,
        )
        result = pl.read_csv(out, separator="\t")
        assert len(result) == len(TEST_PARTICIPANT_IDS)


# ---------------------------------------------------------------------------
# Hail vs VCF equivalence tests
# ---------------------------------------------------------------------------

# CFTR deletion — known multiallelic site verified to produce identical
# results from both Hail and VCF pipelines (414,696 records, 0 differences).
_CFTR_CHR = 7
_CFTR_POS = 117559590
_CFTR_REF = "ATCT"
_CFTR_ALT = "A"
_CFTR_GT_DICT = {0: "0/0", 1: ["0/1", "1/1"]}


@pytest.mark.aou
class TestByGenotypeHailVsVcf:
    """Run by_genotype with both hail and vcf for the same variant, compare outputs."""

    @staticmethod
    def _run_and_load(cohort, tmp_path, data_format, chrom, pos, ref, alt, gt_dict):
        out = str(tmp_path / f"{data_format}_cohort.tsv")
        cohort.by_genotype(
            chromosome_number=chrom,
            genomic_position=pos,
            ref_allele=ref,
            alt_allele=alt,
            gt_dict=gt_dict,
            data_format=data_format,
            output_file_path=out,
        )
        return pl.read_csv(out, separator="\t").sort("person_id")

    @staticmethod
    def _compare(hail_df, vcf_df):
        """Compare two cohort DataFrames and return diagnostics."""
        hail_ids = set(hail_df["person_id"].to_list())
        vcf_ids = set(vcf_df["person_id"].to_list())

        only_hail = hail_ids - vcf_ids
        only_vcf = vcf_ids - hail_ids
        common = hail_ids & vcf_ids

        hail_common = hail_df.filter(pl.col("person_id").is_in(list(common)))
        vcf_common = vcf_df.filter(pl.col("person_id").is_in(list(common)))
        merged = hail_common.join(vcf_common, on="person_id", suffix="_vcf")
        mismatched = merged.filter(pl.col("genotype") != pl.col("genotype_vcf"))

        return {
            "only_hail": only_hail,
            "only_vcf": only_vcf,
            "mismatched": mismatched,
        }

    def test_cftr_hail_and_vcf_match(self, tmp_path):
        cohort = Cohort(platform="aou", aou_db_version=8)
        hail_df = self._run_and_load(
            cohort, tmp_path, "hail",
            _CFTR_CHR, _CFTR_POS, _CFTR_REF, _CFTR_ALT, _CFTR_GT_DICT,
        )
        vcf_df = self._run_and_load(
            cohort, tmp_path, "vcf",
            _CFTR_CHR, _CFTR_POS, _CFTR_REF, _CFTR_ALT, _CFTR_GT_DICT,
        )

        diffs = self._compare(hail_df, vcf_df)

        assert len(diffs["only_hail"]) == 0, (
            f"Samples only in Hail: {sorted(diffs['only_hail'])[:20]}"
        )
        assert len(diffs["only_vcf"]) == 0, (
            f"Samples only in VCF: {sorted(diffs['only_vcf'])[:20]}"
        )
        assert len(diffs["mismatched"]) == 0, (
            f"Genotype mismatches:\n{diffs['mismatched'].head(20)}"
        )
        assert len(hail_df) == len(vcf_df), (
            f"Record count mismatch: Hail={len(hail_df)}, VCF={len(vcf_df)}"
        )
