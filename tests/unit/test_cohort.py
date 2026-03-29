"""
Unit tests for cohort.py — BigQuery and Hail calls are mocked.
"""
import os
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

    def test_aou_omop_cdr_overrides_env(self, aou_env):
        c = Cohort(platform="aou", aou_db_version=8, aou_omop_cdr="override.cdr")
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
