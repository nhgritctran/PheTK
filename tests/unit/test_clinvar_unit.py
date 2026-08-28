"""Unit tests for the clinvar module. No network access."""

import polars as pl
import pytest
from unittest.mock import patch

from phetk import clinvar
from phetk.clinvar import (
    ClinVar,
    REVIEW_STATUS_STARS,
    available_clinical_significances,
    available_review_statuses,
    _norm_info,
    _parse_gene_info,
    _parse_molecular_consequence,
    _parse_region,
    _significance_components,
)


# ---------------------------------------------------------------------------
# Fake pysam objects
# ---------------------------------------------------------------------------

class FakeRecord:
    """Minimal stand-in for pysam.VariantRecord."""

    def __init__(self, chrom="7", pos=100, ref="C", alts=("T",), variation_id="12345", **info):
        self.chrom = chrom
        self.pos = pos
        self.ref = ref
        self.alts = alts
        self.id = variation_id
        self.info = info


class FakeVariantFile:
    """Minimal stand-in for pysam.VariantFile with a fixed record set."""

    def __init__(self, records, contigs=("7",)):
        self._records = records
        self._contigs = set(contigs)

    def fetch(self, contig, start, end):
        if contig not in self._contigs:
            raise ValueError(f"invalid contig {contig}")
        return iter(self._records)


def make_clinvar(records, contigs=("7",), **kwargs):
    """Build a ClinVar instance whose VCF handle is a FakeVariantFile."""
    instance = ClinVar(**kwargs)
    instance._vcf = FakeVariantFile(records, contigs=contigs)
    return instance


def search(clinvar_obj, tmp_path, **kwargs):
    """Run a search writing output into tmp_path."""
    kwargs.setdefault("output_file_path", str(tmp_path / "out.tsv"))
    return clinvar_obj.search(**kwargs)


# ---------------------------------------------------------------------------
# Compound CLNSIG handling -- the likeliest correctness bug
# ---------------------------------------------------------------------------

class TestCompoundSignificance:

    @pytest.mark.parametrize("value, expected", [
        ("Pathogenic/Likely pathogenic", {"pathogenic", "likely pathogenic",
                                          "pathogenic/likely pathogenic"}),
        ("Benign/Likely benign", {"benign", "likely benign", "benign/likely benign"}),
        ("Pathogenic|drug response", {"pathogenic", "drug response",
                                      "pathogenic|drug response"}),
        ("Uncertain significance/Uncertain risk allele",
         {"uncertain significance", "uncertain risk allele",
          "uncertain significance/uncertain risk allele"}),
        ("Uncertain significance", {"uncertain significance"}),
    ])
    def test_components(self, value, expected):
        assert _significance_components(value) == expected

    def test_conflicting_survives_intact(self):
        value = "Conflicting classifications of pathogenicity"
        assert _significance_components(value) == {value.lower()}

    def test_none_and_empty(self):
        assert _significance_components(None) == set()
        assert _significance_components("") == set()

    def test_pathogenic_filter_keeps_compound(self, tmp_path):
        records = [
            FakeRecord(pos=1, CLNSIG=("Pathogenic/Likely_pathogenic",)),
            FakeRecord(pos=2, CLNSIG=("Pathogenic|drug_response",)),
            FakeRecord(pos=3, CLNSIG=("Pathogenic",)),
            FakeRecord(pos=4, CLNSIG=("Benign",)),
            FakeRecord(pos=5, CLNSIG=("Conflicting_classifications_of_pathogenicity",)),
        ]
        df = search(make_clinvar(records), tmp_path,
                    region="7:1-100", clinical_significance="Pathogenic")
        assert df["genomic_position"].to_list() == [1, 2, 3]

    def test_conflicting_filter_is_exact(self, tmp_path):
        records = [
            FakeRecord(pos=1, CLNSIG=("Pathogenic/Likely_pathogenic",)),
            FakeRecord(pos=2, CLNSIG=("Conflicting_classifications_of_pathogenicity",)),
        ]
        df = search(make_clinvar(records), tmp_path, region="7:1-100",
                    clinical_significance="Conflicting classifications of pathogenicity")
        assert df["genomic_position"].to_list() == [2]

    def test_multiple_significances(self, tmp_path):
        records = [
            FakeRecord(pos=1, CLNSIG=("Benign/Likely_benign",)),
            FakeRecord(pos=2, CLNSIG=("Uncertain_significance",)),
            FakeRecord(pos=3, CLNSIG=("Pathogenic",)),
        ]
        df = search(make_clinvar(records), tmp_path, region="7:1-100",
                    clinical_significance=["Likely benign", "Pathogenic"])
        assert df["genomic_position"].to_list() == [1, 3]

    def test_missing_clnsig_does_not_crash(self, tmp_path):
        records = [FakeRecord(pos=1), FakeRecord(pos=2, CLNSIG=None)]
        df = search(make_clinvar(records), tmp_path, region="7:1-100")
        assert len(df) == 2
        assert df["clinical_significance"].to_list() == [None, None]

    def test_missing_clnsig_excluded_by_filter(self, tmp_path):
        records = [FakeRecord(pos=1, CLNSIG=None), FakeRecord(pos=2, CLNSIG=("Pathogenic",))]
        df = search(make_clinvar(records), tmp_path, region="7:1-100",
                    clinical_significance="Pathogenic")
        assert df["genomic_position"].to_list() == [2]

    def test_unknown_significance_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Unsupported clinical_significance"):
            search(make_clinvar([]), tmp_path, region="7:1-100",
                   clinical_significance="VUS-high")


# ---------------------------------------------------------------------------
# pysam tuple / underscore reassembly
# ---------------------------------------------------------------------------

class TestNormInfo:

    def test_pysam_tuple_split_on_comma_is_reassembled(self):
        raw = ("criteria_provided", "_single_submitter")
        assert _norm_info(raw) == "criteria provided, single submitter"

    def test_multiple_submitters_no_conflicts(self):
        raw = ("criteria_provided", "_multiple_submitters", "_no_conflicts")
        assert _norm_info(raw) == "criteria provided, multiple submitters, no conflicts"

    def test_single_element_tuple(self):
        assert _norm_info(("Likely_benign",)) == "Likely benign"

    def test_plain_string(self):
        assert _norm_info("single_nucleotide_variant") == "single nucleotide variant"

    def test_none_empty_tuple_and_blank(self):
        assert _norm_info(None) is None
        assert _norm_info(()) is None
        assert _norm_info("") is None
        assert _norm_info("   ") is None

    def test_reassembled_status_maps_to_a_star(self):
        status = _norm_info(("criteria_provided", "_single_submitter"))
        assert REVIEW_STATUS_STARS[status] == 1


class TestFieldParsers:

    def test_molecular_consequence_strips_so_term(self):
        raw = ("SO:0001583|missense_variant",)
        assert _parse_molecular_consequence(raw) == "missense variant"

    def test_molecular_consequence_dedupes(self):
        raw = ("SO:0001583|missense_variant", "SO:0001583|missense_variant",
               "SO:0001589|frameshift_variant")
        assert _parse_molecular_consequence(raw) == "missense variant,frameshift variant"

    def test_molecular_consequence_none(self):
        assert _parse_molecular_consequence(None) is None

    def test_gene_info_strips_gene_ids(self):
        assert _parse_gene_info("CFTR:1080|LOC111674463:111674463") == "CFTR|LOC111674463"

    def test_gene_info_single_gene(self):
        assert _parse_gene_info("BRCA1:672") == "BRCA1"

    def test_gene_info_none(self):
        assert _parse_gene_info(None) is None


# ---------------------------------------------------------------------------
# Review status star mapping
# ---------------------------------------------------------------------------

class TestReviewStars:

    @pytest.mark.parametrize("status, star", [
        ("practice guideline", 4),
        ("reviewed by expert panel", 3),
        ("criteria provided, multiple submitters, no conflicts", 2),
        ("criteria provided, single submitter", 1),
        ("criteria provided, conflicting classifications", 1),
        ("no assertion criteria provided", 0),
        ("no classification provided", 0),
        ("no classification for the single variant", 0),
        ("no classifications from unflagged records", 0),
    ])
    def test_all_nine_statuses(self, status, star):
        assert REVIEW_STATUS_STARS[status] == star

    def test_vocabulary_is_complete(self):
        assert len(REVIEW_STATUS_STARS) == 9
        assert set(available_review_statuses()) == set(REVIEW_STATUS_STARS)

    def test_available_review_statuses_ordered_by_star_desc(self):
        stars = [REVIEW_STATUS_STARS[s] for s in available_review_statuses()]
        assert stars == sorted(stars, reverse=True)

    def test_available_clinical_significances_is_a_copy(self):
        first = available_clinical_significances()
        first.append("mutated")
        assert "mutated" not in available_clinical_significances()

    def test_star_column_derived(self, tmp_path):
        records = [
            FakeRecord(pos=1, CLNREVSTAT=("practice_guideline",)),
            FakeRecord(pos=2, CLNREVSTAT=("criteria_provided", "_single_submitter")),
            FakeRecord(pos=3, CLNREVSTAT=("no_assertion_criteria_provided",)),
        ]
        df = search(make_clinvar(records), tmp_path, region="7:1-100")
        assert df["review_star"].to_list() == [4, 1, 0]

    def test_min_review_star_filters(self, tmp_path):
        records = [
            FakeRecord(pos=1, CLNREVSTAT=("practice_guideline",)),
            FakeRecord(pos=2, CLNREVSTAT=("criteria_provided", "_single_submitter")),
            FakeRecord(pos=3, CLNREVSTAT=("no_assertion_criteria_provided",)),
        ]
        df = search(make_clinvar(records), tmp_path, region="7:1-100", min_review_star=2)
        assert df["genomic_position"].to_list() == [1]

    def test_min_review_star_none_keeps_zero_star(self, tmp_path):
        records = [FakeRecord(pos=1, CLNREVSTAT=("no_assertion_criteria_provided",))]
        df = search(make_clinvar(records), tmp_path, region="7:1-100")
        assert len(df) == 1

    def test_missing_review_status_gets_null_star(self, tmp_path):
        df = search(make_clinvar([FakeRecord(pos=1)]), tmp_path, region="7:1-100")
        assert df["review_star"].to_list() == [None]
        assert df["review_status"].to_list() == [None]

    def test_missing_review_status_dropped_by_min_star_zero(self, tmp_path):
        records = [
            FakeRecord(pos=1),
            FakeRecord(pos=2, CLNREVSTAT=("no_assertion_criteria_provided",)),
        ]
        df = search(make_clinvar(records), tmp_path, region="7:1-100", min_review_star=0)
        assert df["genomic_position"].to_list() == [2]

    def test_review_status_filter(self, tmp_path):
        records = [
            FakeRecord(pos=1, CLNREVSTAT=("criteria_provided", "_single_submitter")),
            FakeRecord(pos=2, CLNREVSTAT=("reviewed_by_expert_panel",)),
        ]
        df = search(make_clinvar(records), tmp_path, region="7:1-100",
                    review_status="reviewed by expert panel")
        assert df["genomic_position"].to_list() == [2]

    def test_unknown_review_status_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Unsupported review_status"):
            search(make_clinvar([]), tmp_path, region="7:1-100", review_status="four stars")

    @pytest.mark.parametrize("value", [-1, 5])
    def test_out_of_range_min_star_raises(self, value, tmp_path):
        with pytest.raises(ValueError, match="min_review_star must be between 0 and 4"):
            search(make_clinvar([]), tmp_path, region="7:1-100", min_review_star=value)


# ---------------------------------------------------------------------------
# gene / region argument handling
# ---------------------------------------------------------------------------

class TestGeneRegionArguments:

    def test_both_given_raises(self, tmp_path):
        with pytest.raises(ValueError, match="exactly one of gene or region"):
            search(make_clinvar([]), tmp_path, gene="CFTR", region="7:1-100")

    def test_neither_given_raises(self, tmp_path):
        with pytest.raises(ValueError, match="exactly one of gene or region"):
            search(make_clinvar([]), tmp_path)

    def test_gene_path_uses_lookup(self, tmp_path):
        instance = make_clinvar([FakeRecord(pos=117480082)])
        with patch("phetk.clinvar.get_gene_region",
                   return_value=("7", 117480025, 117668665)) as lookup:
            df = search(instance, tmp_path, gene="CFTR")
        lookup.assert_called_once_with("CFTR", assembly="GRCh38")
        assert len(df) == 1

    @pytest.mark.parametrize("region, expected", [
        ("chr7:117480025-117668665", ("7", 117480025, 117668665)),
        ("7:117480025-117668665", ("7", 117480025, 117668665)),
        ("CHR7:117480025-117668665", ("7", 117480025, 117668665)),
        ("chrX:1-100", ("X", 1, 100)),
        ("X:1-100", ("X", 1, 100)),
        ("chrY:1-100", ("Y", 1, 100)),
        ("MT:1-100", ("MT", 1, 100)),
        ("7:117,480,025-117,668,665", ("7", 117480025, 117668665)),
        ("chr7: 1 - 100", ("7", 1, 100)),
    ])
    def test_region_parsing(self, region, expected):
        assert _parse_region(region) == expected

    @pytest.mark.parametrize("region", ["CFTR", "7-100", "7:abc-100", "chr7:100", ""])
    def test_bad_region_raises(self, region):
        with pytest.raises(ValueError, match="Could not parse region"):
            _parse_region(region)

    def test_reversed_region_raises(self):
        with pytest.raises(ValueError, match="greater than region end"):
            _parse_region("7:200-100")

    def test_fetch_converts_to_zero_based_half_open(self, tmp_path):
        instance = make_clinvar([])
        with patch.object(instance._vcf, "fetch", wraps=instance._vcf.fetch) as fetch:
            search(instance, tmp_path, region="7:101-200")
        fetch.assert_called_once_with("7", 100, 200)

    def test_chr_prefixed_contig_in_vcf(self, tmp_path):
        instance = make_clinvar([FakeRecord(chrom="chr7", pos=5)], contigs=("chr7",))
        df = search(instance, tmp_path, region="7:1-100")
        assert df["chromosome_number"].to_list() == ["7"]

    def test_unknown_contig_raises(self, tmp_path):
        with pytest.raises(ValueError, match="is not present in the ClinVar VCF"):
            search(make_clinvar([], contigs=("7",)), tmp_path, region="99:1-100")


# ---------------------------------------------------------------------------
# Output schema and row construction
# ---------------------------------------------------------------------------

class TestOutput:

    def test_full_row(self, tmp_path):
        record = FakeRecord(
            chrom="7", pos=117480082, ref="C", alts=("A",), variation_id="1580358",
            CLNSIG=("Uncertain_significance",),
            CLNREVSTAT=("criteria_provided", "_single_submitter"),
            CLNVC="single_nucleotide_variant",
            GENEINFO="CFTR:1080|LOC111674463:111674463",
            MC=("SO:0001623|5_prime_UTR_variant",),
            CLNDN=("Cystic_fibrosis",),
            RS=("902914688",),
            AF_ESP=0.001, AF_EXAC=0.002, AF_TGP=0.003,
        )
        row = search(make_clinvar([record]), tmp_path, region="7:1-200000000").row(0, named=True)
        assert row == {
            "variant_id": "7:117480082:C:A",
            "rsid": "rs902914688",
            "chromosome_number": "7",
            "genomic_position": 117480082,
            "ref_allele": "C",
            "alt_allele": "A",
            "clinical_significance": "Uncertain significance",
            "review_status": "criteria provided, single submitter",
            "review_star": 1,
            "variation_id": "1580358",
            "gene": "CFTR|LOC111674463",
            "variant_type": "single nucleotide variant",
            "molecular_consequence": "5 prime UTR variant",
            "condition": "Cystic fibrosis",
            "af_esp": 0.001,
            "af_exac": 0.002,
            "af_tgp": 0.003,
        }

    def test_null_rsid(self, tmp_path):
        records = [FakeRecord(pos=1), FakeRecord(pos=2, RS=("12345",))]
        df = search(make_clinvar(records), tmp_path, region="7:1-100")
        assert df["rsid"].to_list() == [None, "rs12345"]
        assert df["rsid"].null_count() == 1

    def test_multi_allelic_split_one_row_per_alt(self, tmp_path):
        record = FakeRecord(pos=50, ref="C", alts=("T", "G"), CLNSIG=("Pathogenic",))
        df = search(make_clinvar([record]), tmp_path, region="7:1-100")
        assert len(df) == 2
        assert df["alt_allele"].to_list() == ["T", "G"]
        assert df["variant_id"].to_list() == ["7:50:C:T", "7:50:C:G"]
        assert df["clinical_significance"].to_list() == ["Pathogenic"] * 2

    def test_empty_result_has_full_typed_schema(self, tmp_path):
        df = search(make_clinvar([]), tmp_path, region="7:1-100")
        assert len(df) == 0
        assert df.schema == pl.Schema(clinvar._OUTPUT_SCHEMA)

    def test_schema_stable_with_rows(self, tmp_path):
        df = search(make_clinvar([FakeRecord()]), tmp_path, region="7:1-100")
        assert df.schema == pl.Schema(clinvar._OUTPUT_SCHEMA)

    def test_chromosome_is_string_for_x_and_y(self, tmp_path):
        instance = make_clinvar([FakeRecord(chrom="X", pos=5)], contigs=("X",))
        df = search(instance, tmp_path, region="X:1-100")
        assert df["chromosome_number"].dtype == pl.Utf8
        assert df["chromosome_number"].to_list() == ["X"]

    def test_writes_tsv_file(self, tmp_path):
        path = tmp_path / "variants.tsv"
        search(make_clinvar([FakeRecord()]), tmp_path, region="7:1-100",
               output_file_path=str(path))
        assert path.exists()
        assert pl.read_csv(path, separator="\t").height == 1

    def test_default_output_filename(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        make_clinvar([FakeRecord()]).search(region="7:1-100")
        assert (tmp_path / "clinvar_7_1_100_GRCh38.tsv").exists()

    def test_default_output_filename_for_gene(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        instance = make_clinvar([FakeRecord()])
        with patch("phetk.clinvar.get_gene_region", return_value=("7", 1, 100)):
            instance.search(gene="CFTR")
        assert (tmp_path / "clinvar_CFTR_GRCh38.tsv").exists()


# ---------------------------------------------------------------------------
# Remaining filters
# ---------------------------------------------------------------------------

class TestFilters:

    def test_variant_type(self, tmp_path):
        records = [
            FakeRecord(pos=1, CLNVC="single_nucleotide_variant"),
            FakeRecord(pos=2, CLNVC="Deletion"),
            FakeRecord(pos=3, CLNVC="Duplication"),
        ]
        df = search(make_clinvar(records), tmp_path, region="7:1-100",
                    variant_type=["Deletion", "Duplication"])
        assert df["genomic_position"].to_list() == [2, 3]

    @pytest.mark.parametrize("requested", ["SNV", "snv", "single nucleotide variant",
                                           "single_nucleotide_variant"])
    def test_variant_type_snv_aliases(self, requested, tmp_path):
        records = [
            FakeRecord(pos=1, CLNVC="single_nucleotide_variant"),
            FakeRecord(pos=2, CLNVC="Deletion"),
        ]
        df = search(make_clinvar(records), tmp_path, region="7:1-100", variant_type=requested)
        assert df["genomic_position"].to_list() == [1]

    def test_max_allele_frequency(self, tmp_path):
        records = [
            FakeRecord(pos=1, AF_ESP=0.5),
            FakeRecord(pos=2, AF_ESP=0.0001),
            FakeRecord(pos=3, AF_TGP=0.2, AF_ESP=0.00001),
            FakeRecord(pos=4),
        ]
        df = search(make_clinvar(records), tmp_path, region="7:1-100",
                    max_allele_frequency=0.01)
        # pos=3 is dropped because the *highest* reported frequency wins;
        # pos=4 has no reported frequency and is kept.
        assert df["genomic_position"].to_list() == [2, 4]

    def test_filters_combine(self, tmp_path):
        records = [
            FakeRecord(pos=1, CLNSIG=("Uncertain_significance",),
                       CLNREVSTAT=("criteria_provided", "_multiple_submitters", "_no_conflicts")),
            FakeRecord(pos=2, CLNSIG=("Uncertain_significance",),
                       CLNREVSTAT=("criteria_provided", "_single_submitter")),
            FakeRecord(pos=3, CLNSIG=("Pathogenic",),
                       CLNREVSTAT=("reviewed_by_expert_panel",)),
        ]
        df = search(make_clinvar(records), tmp_path, region="7:1-100",
                    clinical_significance="Uncertain significance", min_review_star=2)
        assert df["genomic_position"].to_list() == [1]


# ---------------------------------------------------------------------------
# Constructor and error style
# ---------------------------------------------------------------------------

class TestConstructor:

    @pytest.mark.parametrize("given, expected", [
        ("GRCh38", "GRCh38"), ("grch38", "GRCh38"), ("hg38", "GRCh38"),
        ("GRCh37", "GRCh37"), ("grch37", "GRCh37"), ("hg19", "GRCh37"),
        (" GRCh38 ", "GRCh38"),
    ])
    def test_assembly_normalization(self, given, expected):
        assert ClinVar(assembly=given).assembly == expected

    def test_unsupported_assembly_raises(self):
        with pytest.raises(ValueError, match="Unsupported assembly"):
            ClinVar(assembly="GRCh36")

    def test_default_vcf_url_follows_assembly(self):
        assert ClinVar().vcf_path.endswith("/vcf_GRCh38/clinvar.vcf.gz")
        assert ClinVar(assembly="GRCh37").vcf_path.endswith("/vcf_GRCh37/clinvar.vcf.gz")

    def test_vcf_path_override(self):
        assert ClinVar(vcf_path="gs://bucket/clinvar.vcf.gz").vcf_path == "gs://bucket/clinvar.vcf.gz"

    def test_unopenable_vcf_raises_value_error(self):
        instance = ClinVar(vcf_path="/nonexistent/clinvar.vcf.gz")
        with pytest.raises(ValueError, match="Could not open ClinVar VCF"):
            instance._open_vcf()


class TestRemoteIndexHandling:
    """
    Every ClinVar release is named clinvar.vcf.gz, and htslib caches a remote
    index into the working directory under that basename. Left alone, the
    GRCh38 and GRCh37 indexes collide and the second assembly opened in a given
    directory fails with "Invalid BGZF header". The index is fetched explicitly
    to a URL-derived path to prevent that.
    """

    def setup_method(self):
        clinvar._INDEX_CACHE.clear()

    def teardown_method(self):
        clinvar._INDEX_CACHE.clear()

    def test_remote_vcf_gets_explicit_index(self):
        instance = ClinVar()
        with patch("phetk.clinvar._fetch_remote_index", return_value="/tmp/abc.tbi") as fetch, \
                patch("pysam.VariantFile") as variant_file:
            instance._open_vcf()
        fetch.assert_called_once_with(instance.vcf_path)
        assert variant_file.call_args.kwargs["index_filename"] == "/tmp/abc.tbi"

    @pytest.mark.parametrize("path", ["/data/clinvar.vcf.gz", "gs://bucket/clinvar.vcf.gz"])
    def test_local_and_gcs_paths_use_the_sibling_index(self, path):
        instance = ClinVar(vcf_path=path)
        with patch("phetk.clinvar._fetch_remote_index") as fetch, \
                patch("pysam.VariantFile") as variant_file:
            instance._open_vcf()
        fetch.assert_not_called()
        assert variant_file.call_args.kwargs["index_filename"] is None

    def test_assemblies_get_distinct_index_paths(self, monkeypatch):
        downloaded = []

        def fake_urlopen(url, timeout=None):
            downloaded.append(url)
            from io import BytesIO
            payload = BytesIO(b"fake index")
            payload.__enter__ = lambda s=payload: s
            payload.__exit__ = lambda s, *a: None
            return payload

        monkeypatch.setattr("phetk.clinvar.urllib.request.urlopen", fake_urlopen)
        grch38 = clinvar._fetch_remote_index(ClinVar("GRCh38").vcf_path)
        grch37 = clinvar._fetch_remote_index(ClinVar("GRCh37").vcf_path)

        assert grch38 != grch37, "GRCh38 and GRCh37 indexes must not share a local path"
        assert downloaded == [
            "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz.tbi",
            "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh37/clinvar.vcf.gz.tbi",
        ]

    def test_index_downloaded_once_per_url(self, monkeypatch):
        calls = []

        def fake_urlopen(url, timeout=None):
            calls.append(url)
            from io import BytesIO
            payload = BytesIO(b"fake index")
            payload.__enter__ = lambda s=payload: s
            payload.__exit__ = lambda s, *a: None
            return payload

        monkeypatch.setattr("phetk.clinvar.urllib.request.urlopen", fake_urlopen)
        url = ClinVar().vcf_path
        assert clinvar._fetch_remote_index(url) == clinvar._fetch_remote_index(url)
        assert len(calls) == 1

    def test_index_is_not_written_to_the_working_directory(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)

        def fake_urlopen(url, timeout=None):
            from io import BytesIO
            payload = BytesIO(b"fake index")
            payload.__enter__ = lambda s=payload: s
            payload.__exit__ = lambda s, *a: None
            return payload

        monkeypatch.setattr("phetk.clinvar.urllib.request.urlopen", fake_urlopen)
        clinvar._fetch_remote_index(ClinVar().vcf_path)
        assert list(tmp_path.iterdir()) == []

    def test_index_download_failure_raises_value_error(self, monkeypatch):
        def boom(url, timeout=None):
            raise OSError("connection refused")

        monkeypatch.setattr("phetk.clinvar.urllib.request.urlopen", boom)
        with pytest.raises(ValueError, match="Could not download tabix index"):
            clinvar._fetch_remote_index(ClinVar().vcf_path)


class TestByGenotypeHandoff:
    """
    The four locus columns are named after Cohort.by_genotype()'s parameters so
    a row splats straight in. If either side is renamed, these fail.
    """

    LOCUS_COLUMNS = ["chromosome_number", "genomic_position", "ref_allele", "alt_allele"]

    def test_column_names_match_by_genotype_parameters(self):
        import inspect
        from phetk.cohort import Cohort

        parameters = inspect.signature(Cohort.by_genotype).parameters
        for column in self.LOCUS_COLUMNS:
            assert column in parameters, (
                f"{column!r} is not a Cohort.by_genotype() parameter; "
                f"the ClinVar output schema and by_genotype have drifted apart."
            )

    def test_locus_columns_present_in_output(self, tmp_path):
        df = search(make_clinvar([FakeRecord()]), tmp_path, region="7:1-100")
        assert set(self.LOCUS_COLUMNS).issubset(df.columns)

    def test_row_splats_into_by_genotype(self, tmp_path):
        from phetk.cohort import Cohort

        record = FakeRecord(chrom="7", pos=117480099, ref="A", alts=("C",))
        df = search(make_clinvar([record]), tmp_path, region="7:1-200000000")
        row = df.select(self.LOCUS_COLUMNS).row(0, named=True)

        with patch.object(Cohort, "__init__", return_value=None), \
                patch.object(Cohort, "by_genotype", return_value=None) as by_genotype:
            Cohort().by_genotype(**row)

        by_genotype.assert_called_once_with(
            chromosome_number="7", genomic_position=117480099,
            ref_allele="A", alt_allele="C",
        )

    def test_chromosome_stays_unprefixed_for_by_genotype(self, tmp_path):
        # by_genotype prepends "chr" itself on GRCh38 (cohort.py:206), so a
        # "chr7" value here would produce "chrchr7".
        df = search(make_clinvar([FakeRecord(chrom="chr7")], contigs=("chr7",)),
                    tmp_path, region="7:1-100")
        assert df["chromosome_number"].to_list() == ["7"]


class TestErrorStyle:
    """Core API raises ValueError; only the CLI wrapper calls sys.exit."""

    def test_public_api_raises_value_error_not_system_exit(self, tmp_path):
        with pytest.raises(ValueError):
            search(make_clinvar([]), tmp_path)
        with pytest.raises(ValueError):
            ClinVar(assembly="nope")

    def test_cli_wrapper_exits_with_code_1(self, monkeypatch, capsys):
        monkeypatch.setattr("sys.argv", ["phetk-clinvar-search", "--gene", "CFTR",
                                         "--region", "7:1-100"])
        with pytest.raises(SystemExit) as exit_info:
            clinvar.main_search()
        assert exit_info.value.code == 1
        assert "exactly one of gene or region" in capsys.readouterr().out


class TestGeneRegionLookup:
    """get_gene_region parsing, with the network call mocked out."""

    SUMMARY = {
        "result": {
            "1080": {
                "name": "CFTR",
                "chromosome_number": "7",
                "genomicinfo": [{"chrloc": "7", "chraccver": "NC_000007.14",
                                 "chrstart": 117480024, "chrstop": 117668664}],
                "locationhist": [
                    {"assemblyaccver": "GCF_000001405.40", "chrloc": "7",
                     "chrstart": 117480024, "chrstop": 117668664},
                    {"assemblyaccver": "GCF_000001405.25", "chrloc": "7",
                     "chrstart": 117120078, "chrstop": 117308718},
                    {"assemblyaccver": "GCF_009914755.1", "chrloc": "7",
                     "chrstart": 118795360, "chrstop": 118984025},
                ],
            }
        }
    }

    @staticmethod
    def fake_eutils(summary=None, idlist=("1080",)):
        summary = summary if summary is not None else TestGeneRegionLookup.SUMMARY

        def _call(endpoint, params, attempts=4):
            if endpoint == "esearch":
                return {"esearchresult": {"idlist": list(idlist)}}
            return summary

        return _call

    def setup_method(self):
        clinvar.get_gene_region.cache_clear()

    def teardown_method(self):
        clinvar.get_gene_region.cache_clear()

    def test_grch38_is_one_based_inclusive(self):
        with patch("phetk.clinvar._eutils_json", side_effect=self.fake_eutils()):
            assert clinvar.get_gene_region("CFTR") == ("7", 117480025, 117668665)

    def test_grch37_from_location_history(self):
        with patch("phetk.clinvar._eutils_json", side_effect=self.fake_eutils()):
            assert clinvar.get_gene_region("CFTR", assembly="GRCh37") == ("7", 117120079, 117308719)

    def test_minus_strand_coordinates_are_ordered(self):
        summary = {"result": {"1080": {
            "chromosome_number": "17",
            "genomicinfo": [{"chrloc": "17", "chrstart": 43170244, "chrstop": 43044294}],
        }}}
        with patch("phetk.clinvar._eutils_json", side_effect=self.fake_eutils(summary)):
            assert clinvar.get_gene_region("BRCA1") == ("17", 43044295, 43170245)

    def test_result_is_cached(self):
        with patch("phetk.clinvar._eutils_json", side_effect=self.fake_eutils()) as call:
            clinvar.get_gene_region("CFTR")
            clinvar.get_gene_region("CFTR")
        assert call.call_count == 2  # one esearch + one esummary, not four

    def test_unknown_gene_raises(self):
        with patch("phetk.clinvar._eutils_json", side_effect=self.fake_eutils(idlist=())):
            with pytest.raises(ValueError, match="No human gene found"):
                clinvar.get_gene_region("NOTAGENE")

    def test_missing_assembly_raises(self):
        summary = {"result": {"1080": {
            "chromosome_number": "7",
            "locationhist": [{"assemblyaccver": "GCF_009914755.1", "chrloc": "7",
                              "chrstart": 1, "chrstop": 2}],
        }}}
        with patch("phetk.clinvar._eutils_json", side_effect=self.fake_eutils(summary)):
            with pytest.raises(ValueError, match="no GRCh37 coordinates"):
                clinvar.get_gene_region("CFTR", assembly="GRCh37")

    def test_blank_gene_raises(self):
        with pytest.raises(ValueError, match="non-empty gene symbol"):
            clinvar.get_gene_region("   ")

    @pytest.mark.parametrize("accession, expected", [
        ("GCF_000001405.40", "GRCh38"),
        ("GCF_000001405.26", "GRCh38"),
        ("GCF_000001405.25", "GRCh37"),
        ("GCF_000001405.13", "GRCh37"),
        ("GCF_009914755.1", None),
        ("nonsense", None),
        ("GCF_000001405.x", None),
    ])
    def test_assembly_of_accession(self, accession, expected):
        assert clinvar._assembly_of_accession(accession) == expected
