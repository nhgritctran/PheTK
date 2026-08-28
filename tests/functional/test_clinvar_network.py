"""
Functional tests for the clinvar module against live NCBI services.

These hit ftp.ncbi.nlm.nih.gov (tabix range requests) and eutils.ncbi.nlm.nih.gov,
so they are marked ``network`` and skipped unless PHETK_NETWORK_TESTS is set.

The ClinVar VCF is a weekly snapshot, so every assertion below is a range or a
proportion rather than an exact count.
"""

import polars as pl
import pytest

from phetk.clinvar import ClinVar, REVIEW_STATUS_STARS, get_gene_region

pytestmark = pytest.mark.network

CFTR_CHROM = "7"


@pytest.fixture(scope="module")
def clinvar():
    return ClinVar()


@pytest.fixture(scope="module")
def cftr_vus(clinvar, tmp_path_factory):
    out = tmp_path_factory.mktemp("clinvar") / "cftr_vus.tsv"
    return clinvar.search(
        gene="CFTR",
        clinical_significance="Uncertain significance",
        output_file_path=str(out),
    )


class TestGeneLookup:

    def test_cftr_grch38_region(self):
        chromosome, start, end = get_gene_region("CFTR")
        assert chromosome == CFTR_CHROM
        assert 117_400_000 < start < 117_500_000
        assert 117_600_000 < end < 117_800_000
        assert end > start

    def test_brca1_is_minus_strand_but_ordered(self):
        chromosome, start, end = get_gene_region("BRCA1")
        assert chromosome == "17"
        assert start < end

    def test_grch37_differs_from_grch38(self):
        assert get_gene_region("CFTR", assembly="GRCh37") != get_gene_region("CFTR")

    def test_unknown_gene_raises(self):
        with pytest.raises(ValueError):
            get_gene_region("NOTAREALGENESYMBOL123")


class TestCftrVus:

    def test_expected_variant_count(self, cftr_vus):
        assert 2000 <= len(cftr_vus) <= 6000

    def test_rsid_coverage_above_half(self, cftr_vus):
        coverage = 1 - cftr_vus["rsid"].null_count() / len(cftr_vus)
        assert coverage > 0.5

    def test_all_rsids_are_well_formed(self, cftr_vus):
        rsids = cftr_vus["rsid"].drop_nulls()
        assert rsids.str.starts_with("rs").all()

    def test_all_rows_have_coordinates(self, cftr_vus):
        assert cftr_vus["chromosome_number"].null_count() == 0
        assert cftr_vus["genomic_position"].null_count() == 0
        assert cftr_vus["ref_allele"].null_count() == 0
        assert cftr_vus["alt_allele"].null_count() == 0

    def test_all_rows_are_on_cftr_chromosome_and_in_span(self, cftr_vus):
        _, start, end = get_gene_region("CFTR")
        assert cftr_vus["chromosome_number"].unique().to_list() == [CFTR_CHROM]
        assert cftr_vus["genomic_position"].min() >= start
        assert cftr_vus["genomic_position"].max() <= end

    def test_chromosome_is_unprefixed_string(self, cftr_vus):
        assert cftr_vus["chromosome_number"].dtype == pl.Utf8
        assert not cftr_vus["chromosome_number"].str.starts_with("chr").any()

    def test_every_row_is_a_vus(self, cftr_vus):
        significances = set(cftr_vus["clinical_significance"].to_list())
        assert significances
        for value in significances:
            assert "Uncertain significance" in value

    def test_exactly_four_review_statuses(self, cftr_vus):
        assert set(cftr_vus["review_status"].to_list()) == {
            "criteria provided, single submitter",
            "criteria provided, multiple submitters, no conflicts",
            "no assertion criteria provided",
            "reviewed by expert panel",
        }

    def test_review_stars_match_status(self, cftr_vus):
        pairs = cftr_vus.select(["review_status", "review_star"]).unique()
        for status, star in pairs.iter_rows():
            assert REVIEW_STATUS_STARS[status] == star

    def test_variant_id_matches_components(self, cftr_vus):
        rebuilt = pl.concat_str(
            ["chromosome_number", "genomic_position", "ref_allele", "alt_allele"], separator=":"
        )
        assert cftr_vus.select(rebuilt.eq(pl.col("variant_id")).all()).item()

    def test_cftr_is_among_the_annotated_genes(self, cftr_vus):
        assert cftr_vus["gene"].str.contains("CFTR").all()

    def test_output_file_is_readable(self, clinvar, tmp_path):
        path = tmp_path / "out.tsv"
        df = clinvar.search(region="7:117480025-117490000", output_file_path=str(path))
        assert path.exists()
        assert pl.read_csv(path, separator="\t", infer_schema_length=None).height == len(df)


class TestBothAssemblies:
    """
    Both ClinVar releases are named clinvar.vcf.gz. Querying one assembly then
    the other used to fail with "Invalid BGZF header" because htslib reused the
    first assembly's cached index.
    """

    def test_grch37_then_grch38_in_one_process(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        results = {}
        for assembly in ("GRCh37", "GRCh38"):
            results[assembly] = ClinVar(assembly=assembly).search(
                gene="CFTR",
                clinical_significance="Uncertain significance",
                min_review_star=2,
                output_file_path=str(tmp_path / f"{assembly}.tsv"),
            )

        grch37, grch38 = results["GRCh37"], results["GRCh38"]
        assert len(grch37) > 0 and len(grch38) > 0
        # Same variant set, different coordinates.
        assert set(grch37["rsid"].drop_nulls()) == set(grch38["rsid"].drop_nulls())
        assert grch37["genomic_position"].max() < grch38["genomic_position"].min()

    def test_no_index_written_to_working_directory(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        ClinVar().search(region="7:117480025-117490000",
                         output_file_path=str(tmp_path / "out.tsv"))
        assert not list(tmp_path.glob("*.tbi"))

    def test_grch37_coordinates_differ_from_grch38(self):
        assert get_gene_region("CFTR", assembly="GRCh37")[1] < get_gene_region("CFTR")[1]


class TestFiltersAgainstLiveData:

    def test_min_review_star_narrows_results(self, clinvar, cftr_vus, tmp_path):
        strict = clinvar.search(
            gene="CFTR",
            clinical_significance="Uncertain significance",
            min_review_star=2,
            output_file_path=str(tmp_path / "strict.tsv"),
        )
        assert 0 < len(strict) < len(cftr_vus)
        assert strict["review_star"].min() >= 2

    def test_pathogenic_filter_matches_compound_values(self, clinvar, tmp_path):
        df = clinvar.search(
            gene="CFTR",
            clinical_significance="Pathogenic",
            output_file_path=str(tmp_path / "path.tsv"),
        )
        values = set(df["clinical_significance"].to_list())
        assert "Pathogenic" in values
        assert any("/" in value or "|" in value for value in values)
        assert "Benign" not in values

    def test_variant_type_filter(self, clinvar, tmp_path):
        df = clinvar.search(
            gene="CFTR",
            variant_type="SNV",
            output_file_path=str(tmp_path / "snv.tsv"),
        )
        assert len(df) > 0
        assert df["variant_type"].unique().to_list() == ["single nucleotide variant"]

    def test_region_and_gene_queries_agree(self, clinvar, cftr_vus, tmp_path):
        chromosome, start, end = get_gene_region("CFTR")
        df = clinvar.search(
            region=f"chr{chromosome}:{start}-{end}",
            clinical_significance="Uncertain significance",
            output_file_path=str(tmp_path / "region.tsv"),
        )
        assert len(df) == len(cftr_vus)

    def test_empty_region_returns_typed_zero_row_frame(self, clinvar, tmp_path):
        df = clinvar.search(
            region="7:117480025-117480026",
            clinical_significance="Pathogenic",
            min_review_star=4,
            output_file_path=str(tmp_path / "empty.tsv"),
        )
        assert len(df) == 0
        assert df["chromosome_number"].dtype == pl.Utf8
        assert df["genomic_position"].dtype == pl.Int64
