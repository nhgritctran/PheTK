"""
Integration tests for phecode.py — require AoU Workbench environment.
Skipped automatically outside AoU (WORKSPACE_CDR env var not set).
Run inside AoU with: pytest -m aou
"""
import os
import pytest
import polars as pl

from phetk.phecode import Phecode

aou = pytest.mark.skipif(
    not os.getenv("WORKSPACE_CDR"),
    reason="Requires AoU Workbench environment (WORKSPACE_CDR not set)"
)


@pytest.fixture(scope="module")
def aou_phecode():
    return Phecode(platform="aou")


@pytest.mark.aou
class TestPhecodeAouInit:
    def test_icd_events_not_empty(self, aou_phecode):
        assert len(aou_phecode.icd_events) > 0

    def test_icd_events_has_required_columns(self, aou_phecode):
        for col in ["person_id", "date", "ICD", "vocabulary_id", "flag"]:
            assert col in aou_phecode.icd_events.columns

    def test_flag_values_valid(self, aou_phecode):
        valid_flags = {9, 10, 0}
        flags = set(aou_phecode.icd_events["flag"].unique().to_list())
        assert flags.issubset(valid_flags)

    def test_person_ids_are_integers(self, aou_phecode):
        assert aou_phecode.icd_events["person_id"].dtype in (pl.Int32, pl.Int64)


@pytest.mark.aou
class TestCountPhecodeAou:
    def test_phecodeX_produces_output(self, aou_phecode, tmp_path):
        out = str(tmp_path / "aou_counts.tsv")
        aou_phecode.count_phecode(phecode_version="X", icd_version="US", output_file_path=out)
        assert os.path.exists(out)

    def test_phecodeX_has_required_columns(self, aou_phecode, tmp_path):
        out = str(tmp_path / "aou_counts_cols.tsv")
        aou_phecode.count_phecode(phecode_version="X", icd_version="US", output_file_path=out)
        result = pl.read_csv(out, separator="\t", schema_overrides={"phecode": str})
        for col in ["person_id", "phecode", "count", "first_event_date"]:
            assert col in result.columns

    def test_counts_are_positive(self, aou_phecode, tmp_path):
        out = str(tmp_path / "aou_counts_pos.tsv")
        aou_phecode.count_phecode(phecode_version="X", icd_version="US", output_file_path=out)
        result = pl.read_csv(out, separator="\t", schema_overrides={"phecode": str})
        assert result["count"].min() > 0


@pytest.mark.aou
class TestAddAgeAtFirstEventAou:
    def test_creates_age_column(self, aou_phecode, tmp_path):
        # First generate phecode counts
        counts_out = str(tmp_path / "counts.tsv")
        aou_phecode.count_phecode(phecode_version="X", icd_version="US", output_file_path=counts_out)

        age_out = str(tmp_path / "counts_with_age.tsv")
        aou_phecode.add_age_at_first_event(
            phecode_count_file_path=counts_out,
            output_file_path=age_out,
        )
        result = pl.read_csv(age_out, separator="\t", schema_overrides={"phecode": str})
        assert "age_at_first_event" in result.columns

    def test_age_values_are_reasonable(self, aou_phecode, tmp_path):
        counts_out = str(tmp_path / "counts2.tsv")
        aou_phecode.count_phecode(phecode_version="X", icd_version="US", output_file_path=counts_out)
        age_out = str(tmp_path / "counts_with_age2.tsv")
        aou_phecode.add_age_at_first_event(
            phecode_count_file_path=counts_out,
            output_file_path=age_out,
        )
        result = pl.read_csv(age_out, separator="\t", schema_overrides={"phecode": str})
        ages = result["age_at_first_event"].drop_nulls().to_list()
        assert all(0 < a < 120 for a in ages)
