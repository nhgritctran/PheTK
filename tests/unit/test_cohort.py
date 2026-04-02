"""
Unit tests for cohort.py — BigQuery and Hail calls are mocked.
"""
import os
import warnings
import pytest
import polars as pl
from unittest.mock import patch

from phetk.cohort import Cohort


# ---------------------------------------------------------------------------
# Cohort.__init__
# ---------------------------------------------------------------------------

class TestCohortInit:
    def test_aou_platform_reads_workspace_cdr(self, aou_env):
        c = Cohort(platform="aou", aou_db_version=8)
        assert c.cdr == "test_project.test_cdr"

    def test_aou_platform_reads_google_project(self, aou_env):
        c = Cohort(platform="aou", aou_db_version=8)
        assert c.user_project == "test-project-id"

    def test_gbq_dataset_id_overrides_env_on_aou(self, aou_env):
        c = Cohort(platform="aou", aou_db_version=8, gbq_dataset_id="override.cdr")
        assert c.cdr == "override.cdr"

    def test_aou_platform_stored_lowercase(self, aou_env):
        c = Cohort(platform="AOU", aou_db_version=8)
        assert c.platform == "aou"

    def test_valid_aou_db_versions(self, aou_env):
        for v in [6, 7, 8]:
            c = Cohort(platform="aou", aou_db_version=v)
            assert c.db_version == v

    def test_invalid_aou_db_version_exits(self, aou_env):
        with pytest.raises(SystemExit):
            Cohort(platform="aou", aou_db_version=5)

    def test_custom_platform_sets_cdr_from_gbq_dataset(self):
        c = Cohort(platform="custom", gbq_dataset_id="my_project.my_dataset")
        assert c.cdr == "my_project.my_dataset"

    def test_custom_platform_without_gbq_dataset_exits(self):
        with pytest.raises(SystemExit):
            Cohort(platform="custom")

    def test_invalid_platform_exits(self):
        with pytest.raises(SystemExit):
            Cohort(platform="unknown")

    def test_default_covariate_flags_are_false(self, aou_env):
        c = Cohort(platform="aou")
        assert c.current_age is False
        assert c.sex_at_birth is False
        assert c.genetic_ancestry is False
        assert c.ehr_length is False
        assert c.dx_code_occurrence_count is False

    def test_default_first_n_pcs_is_zero(self, aou_env):
        c = Cohort(platform="aou")
        assert c.first_n_pcs == 0


# ---------------------------------------------------------------------------
# Cohort._get_covariates (mocked BigQuery)
# ---------------------------------------------------------------------------

class TestGetCovariates:
    def test_returns_person_id_only_when_no_flags_set(self, aou_env):
        c = Cohort(platform="aou")
        result = c._get_covariates([1, 2, 3])
        assert list(result.columns) == ["person_id"]
        assert result["person_id"].to_list() == [1, 2, 3]

    def test_current_age_returned_when_requested(self, aou_env, fake_age_df):
        c = Cohort(platform="aou")
        c.current_age = True
        with patch("phetk._utils.polars_gbq", return_value=fake_age_df):
            result = c._get_covariates([1, 2, 3])
        assert "current_age" in result.columns

    def test_current_age_squared_returned(self, aou_env, fake_age_df):
        c = Cohort(platform="aou")
        c.current_age_squared = True
        with patch("phetk._utils.polars_gbq", return_value=fake_age_df):
            result = c._get_covariates([1, 2, 3])
        assert "current_age_squared" in result.columns

    def test_date_of_birth_returned(self, aou_env, fake_age_df):
        c = Cohort(platform="aou")
        c.date_of_birth = True
        with patch("phetk._utils.polars_gbq", return_value=fake_age_df):
            result = c._get_covariates([1, 2, 3])
        assert "date_of_birth" in result.columns

    def test_ehr_length_returned(self, aou_env, fake_ehr_df):
        c = Cohort(platform="aou")
        c.ehr_length = True
        with patch("phetk._utils.polars_gbq", return_value=fake_ehr_df):
            result = c._get_covariates([1, 2, 3])
        assert "ehr_length" in result.columns

    def test_dx_code_occurrence_count_returned(self, aou_env, fake_ehr_df):
        c = Cohort(platform="aou")
        c.dx_code_occurrence_count = True
        with patch("phetk._utils.polars_gbq", return_value=fake_ehr_df):
            result = c._get_covariates([1, 2, 3])
        assert "dx_code_occurrence_count" in result.columns

    def test_sex_at_birth_returned(self, aou_env, fake_sex_df):
        c = Cohort(platform="aou")
        c.sex_at_birth = True
        with patch("phetk._utils.polars_gbq", return_value=fake_sex_df):
            result = c._get_covariates([1, 2, 3])
        assert "sex_at_birth" in result.columns

    def test_multiple_covariates_merged(self, aou_env, fake_age_df, fake_sex_df):
        c = Cohort(platform="aou")
        c.current_age = True
        c.sex_at_birth = True
        with patch("phetk._utils.polars_gbq", side_effect=[fake_age_df, fake_sex_df]):
            result = c._get_covariates([1, 2, 3])
        assert "current_age" in result.columns
        assert "sex_at_birth" in result.columns

    def test_single_participant_id_wrapped(self, aou_env):
        c = Cohort(platform="aou")
        result = c._get_covariates(1)
        assert result["person_id"].to_list() == [1]

    def test_list_participant_ids_accepted(self, aou_env):
        c = Cohort(platform="aou")
        result = c._get_covariates([10, 20, 30])
        assert set(result["person_id"].to_list()) == {10, 20, 30}


# ---------------------------------------------------------------------------
# Cohort.add_covariates (mocked BigQuery)
# ---------------------------------------------------------------------------

class TestAddCovariates:
    def test_none_cohort_file_exits(self, aou_env):
        c = Cohort(platform="aou")
        with pytest.raises(SystemExit):
            c.add_covariates(cohort_file_path=None)

    def test_missing_person_id_column_exits(self, aou_env, tmp_path):
        bad = tmp_path / "bad.tsv"
        bad.write_text("col1\tcol2\n1\t2\n")
        c = Cohort(platform="aou")
        with pytest.raises(SystemExit):
            c.add_covariates(cohort_file_path=str(bad))

    def test_creates_output_file(self, aou_env, cohort_file, fake_age_df, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        out = str(tmp_path / "result.tsv")
        c = Cohort(platform="aou")
        with patch("phetk._utils.polars_gbq", return_value=fake_age_df):
            c.add_covariates(
                cohort_file_path=cohort_file,
                current_age=True,
                output_file_path=out,
            )
        assert os.path.exists(out)

    def test_output_contains_person_id(self, aou_env, cohort_file, fake_age_df, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        out = str(tmp_path / "result.tsv")
        c = Cohort(platform="aou")
        with patch("phetk._utils.polars_gbq", return_value=fake_age_df):
            c.add_covariates(
                cohort_file_path=cohort_file,
                current_age=True,
                output_file_path=out,
            )
        result = pl.read_csv(out, separator="\t")
        assert "person_id" in result.columns

    def test_output_contains_requested_covariate(self, aou_env, cohort_file, fake_age_df, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        out = str(tmp_path / "result.tsv")
        c = Cohort(platform="aou")
        with patch("phetk._utils.polars_gbq", return_value=fake_age_df):
            c.add_covariates(
                cohort_file_path=cohort_file,
                current_age=True,
                output_file_path=out,
            )
        result = pl.read_csv(out, separator="\t")
        assert "current_age" in result.columns

    def test_default_output_path_is_cohort_tsv(self, aou_env, cohort_file, fake_age_df, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        c = Cohort(platform="aou")
        with patch("phetk._utils.polars_gbq", return_value=fake_age_df):
            c.add_covariates(cohort_file_path=cohort_file, current_age=True)
        assert os.path.exists(tmp_path / "cohort.tsv")


# ---------------------------------------------------------------------------
# Cohort._validate_by_genotype_inputs
# ---------------------------------------------------------------------------

class TestValidateByGenotypeInputs:
    def test_default_output_file_path(self):
        gt_dict = {0: "0/0", 1: ["0/1", "1/1"]}
        gt_list, gt_lookup, locus, variant_string, out = \
            Cohort._validate_by_genotype_inputs(1, 12345, "A", "G", gt_dict, "GRCh38", None)
        assert out == "aou_chr1_12345_A_G.tsv"

    def test_custom_output_file_path_preserved(self):
        gt_dict = {0: "0/0", 1: "0/1"}
        _, _, _, _, out = Cohort._validate_by_genotype_inputs(
            1, 100, "A", "G", gt_dict, "GRCh38", "my_output.tsv"
        )
        assert out == "my_output.tsv"

    def test_gt_list_flattened(self):
        gt_dict = {0: "0/0", 1: ["0/1", "1/1"]}
        gt_list, _, _, _, _ = Cohort._validate_by_genotype_inputs(
            1, 100, "A", "G", gt_dict, "GRCh38", None
        )
        assert set(gt_list) == {"0/0", "0/1", "1/1"}

    def test_gt_lookup_built(self):
        gt_dict = {0: "0/0", 1: ["0/1", "1/1"]}
        _, gt_lookup, _, _, _ = Cohort._validate_by_genotype_inputs(
            1, 100, "A", "G", gt_dict, "GRCh38", None
        )
        assert gt_lookup == {"0/0": 0, "0/1": 1, "1/1": 1}

    def test_locus_grch38(self):
        gt_dict = {0: "0/0", 1: "0/1"}
        _, _, locus, _, _ = Cohort._validate_by_genotype_inputs(
            22, 5000, "C", "T", gt_dict, "GRCh38", None
        )
        assert locus == "chr22:5000"

    def test_locus_grch37(self):
        gt_dict = {0: "0/0", 1: "0/1"}
        _, _, locus, _, _ = Cohort._validate_by_genotype_inputs(
            22, 5000, "C", "T", gt_dict, "GRCh37", None
        )
        assert locus == "22:5000"

    def test_variant_string_grch38(self):
        gt_dict = {0: "0/0", 1: "0/1"}
        _, _, _, vs, _ = Cohort._validate_by_genotype_inputs(
            1, 100, "A", "G", gt_dict, "GRCh38", None
        )
        assert vs == "chr1:100:A:G"

    def test_invalid_reference_genome_exits(self):
        gt_dict = {0: "0/0", 1: "0/1"}
        with pytest.raises(SystemExit):
            Cohort._validate_by_genotype_inputs(
                1, 100, "A", "G", gt_dict, "hg19", None
            )

    def test_overlapping_gt_dict_exits(self):
        gt_dict = {0: ["0/0", "0/1"], 1: ["0/1", "1/1"]}
        with pytest.raises(SystemExit):
            Cohort._validate_by_genotype_inputs(
                1, 100, "A", "G", gt_dict, "GRCh38", None
            )


# ---------------------------------------------------------------------------
# Cohort._resolve_data_path
# ---------------------------------------------------------------------------

class TestResolveDataPath:
    def test_data_path_override_returned_as_is(self, aou_env):
        c = Cohort(platform="aou")
        result = c._resolve_data_path("vcf", "acaf_threshold", "/my/path", 1)
        assert result == "/my/path"

    def test_aou_vcf_uses_env_var(self, aou_env, monkeypatch):
        monkeypatch.setenv("WGS_ACAF_THRESHOLD_VCF_PATH", "/env/vcf_dir")
        c = Cohort(platform="aou", aou_db_version=8)
        result = c._resolve_data_path("vcf", "acaf_threshold", None, 1)
        assert result == "/env/vcf_dir"

    def test_aou_vcf_constructs_dir_path(self, aou_env, monkeypatch):
        monkeypatch.delenv("WGS_ACAF_THRESHOLD_VCF_PATH", raising=False)
        c = Cohort(platform="aou", aou_db_version=8)
        result = c._resolve_data_path("vcf", "acaf_threshold", None, 1)
        expected = (
            "gs://fc-aou-datasets-controlled/v8"
            "/wgs/short_read/snpindel/acaf_threshold/vcf/"
        )
        assert result == expected

    def test_aou_vcf_verily_bucket(self, aou_env, monkeypatch):
        monkeypatch.delenv("WGS_ACAF_THRESHOLD_VCF_PATH", raising=False)
        monkeypatch.setenv("GOOGLE_PROJECT", "wb-cold-eggplant-9083")
        c = Cohort(platform="aou", aou_db_version=8)
        result = c._resolve_data_path("vcf", "acaf_threshold", None, 1)
        assert "vwb-aou-datasets-controlled" in result

    def test_aou_vcf_exome_call_set(self, aou_env, monkeypatch):
        monkeypatch.delenv("WGS_ACAF_THRESHOLD_VCF_PATH", raising=False)
        c = Cohort(platform="aou", aou_db_version=8)
        result = c._resolve_data_path("vcf", "exome", None, 22)
        assert "exome" in result
        assert "vcf" in result

    def test_aou_hail_uses_env_var(self, aou_env, monkeypatch):
        monkeypatch.setenv("WGS_ACAF_THRESHOLD_SPLIT_HAIL_PATH", "/env/hail.mt")
        c = Cohort(platform="aou")
        result = c._resolve_data_path("hail", "acaf_threshold", None, 1)
        assert result == "/env/hail.mt"

    def test_aou_hail_missing_env_exits(self, aou_env, monkeypatch):
        monkeypatch.delenv("WGS_ACAF_THRESHOLD_SPLIT_HAIL_PATH", raising=False)
        c = Cohort(platform="aou")
        # Remove the hardcoded fallback so the "no path" branch is reached
        monkeypatch.delattr("phetk._paths.cdr8_mt_path")
        with pytest.raises(SystemExit):
            c._resolve_data_path("hail", "acaf_threshold", None, 1)

    def test_custom_platform_requires_data_path(self):
        c = Cohort(platform="custom", gbq_dataset_id="proj.ds")
        with pytest.raises(SystemExit):
            c._resolve_data_path("vcf", "acaf_threshold", None, 1)

    def test_custom_platform_data_path_returned(self):
        c = Cohort(platform="custom", gbq_dataset_id="proj.ds")
        result = c._resolve_data_path("vcf", "acaf_threshold", "/custom/data", 1)
        assert result == "/custom/data"


# ---------------------------------------------------------------------------
# Cohort._filter_and_map_genotypes
# ---------------------------------------------------------------------------

class TestFilterAndMapGenotypes:
    def test_filters_to_gt_list(self):
        df = pl.DataFrame({
            "person_id": [1, 2, 3, 4],
            "GT": ["0/0", "0/1", "1/1", "0/2"],
        })
        gt_list = ["0/0", "0/1", "1/1"]
        gt_lookup = {"0/0": 0, "0/1": 1, "1/1": 1}
        result = Cohort._filter_and_map_genotypes(df, gt_list, gt_lookup)
        assert len(result) == 3
        assert 4 not in result["person_id"].to_list()

    def test_maps_gt_to_integer_labels(self):
        df = pl.DataFrame({
            "person_id": [1, 2, 3],
            "GT": ["0/0", "0/1", "1/1"],
        })
        gt_list = ["0/0", "0/1", "1/1"]
        gt_lookup = {"0/0": 0, "0/1": 1, "1/1": 1}
        result = Cohort._filter_and_map_genotypes(df, gt_list, gt_lookup)
        assert set(result.columns) == {"person_id", "genotype"}
        row0 = result.filter(pl.col("person_id") == 1)["genotype"][0]
        assert row0 == 0
        row1 = result.filter(pl.col("person_id") == 2)["genotype"][0]
        assert row1 == 1

    def test_deduplicates(self):
        df = pl.DataFrame({
            "person_id": [1, 1],
            "GT": ["0/0", "0/0"],
        })
        gt_list = ["0/0"]
        gt_lookup = {"0/0": 0}
        result = Cohort._filter_and_map_genotypes(df, gt_list, gt_lookup)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Cohort.by_genotype — input validation (no Hail/VCF needed)
# ---------------------------------------------------------------------------

class TestByGenotypeValidation:
    def test_invalid_data_format_exits(self, aou_env):
        c = Cohort(platform="aou")
        with pytest.raises(SystemExit):
            c.by_genotype(
                chromosome_number=1, genomic_position=100,
                ref_allele="A", alt_allele="G",
                gt_dict={0: "0/0", 1: "0/1"},
                data_format="pgen",
            )

    def test_both_mt_path_and_data_path_exits(self, aou_env):
        c = Cohort(platform="aou")
        with pytest.raises(SystemExit):
            c.by_genotype(
                chromosome_number=1, genomic_position=100,
                ref_allele="A", alt_allele="G",
                gt_dict={0: "0/0", 1: "0/1"},
                mt_path="/some/path.mt",
                data_path="/other/path",
            )

    def test_mt_path_emits_deprecation_warning(self, aou_env):
        c = Cohort(platform="aou")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # Mock the extraction method so we don't need Hail/Java
            with patch.object(Cohort, "_extract_genotypes_hail", return_value=None):
                c.by_genotype(
                    chromosome_number=1, genomic_position=100,
                    ref_allele="A", alt_allele="G",
                    gt_dict={0: "0/0", 1: "0/1"},
                    data_format="hail",
                    mt_path="/some/path.mt",
                )
            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) == 1
            assert "mt_path is deprecated" in str(deprecation_warnings[0].message)
