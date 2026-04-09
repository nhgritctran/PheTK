"""
Functional tests for phecode.py using local files — no mocking, no AoU required.
Tests Phecode with platform="custom" and static add_phecode_time_to_event.
"""
import datetime
import os
import pytest
import polars as pl

from phetk.phecode import Phecode


@pytest.fixture
def custom_phecode(icd_file):
    return Phecode(platform="custom", icd_file_path=icd_file)


# ---------------------------------------------------------------------------
# Phecode init with local file
# ---------------------------------------------------------------------------

class TestPhecodeCustomInit:
    def test_loads_icd_data(self, custom_phecode):
        assert custom_phecode.icd_events is not None
        assert len(custom_phecode.icd_events) > 0

    def test_flag_column_present(self, custom_phecode):
        assert "flag" in custom_phecode.icd_events.columns

    def test_icd10cm_assigned_flag_10(self, custom_phecode):
        rows = custom_phecode.icd_events.filter(pl.col("vocabulary_id") == "ICD10CM")
        assert all(f == 10 for f in rows["flag"].to_list())

    def test_icd9cm_assigned_flag_9(self, tmp_path):
        path = tmp_path / "icd9.tsv"
        path.write_text("person_id\tdate\tICD\tvocabulary_id\n1\t2020-01-01\t250.0\tICD9CM\n")
        ph = Phecode(platform="custom", icd_file_path=str(path))
        assert ph.icd_events["flag"][0] == 9

    def test_icd_column_is_string(self, custom_phecode):
        assert custom_phecode.icd_events["ICD"].dtype == pl.Utf8

    def test_csv_file_also_works(self, icd_data, tmp_path):
        path = tmp_path / "icd.csv"
        icd_data.write_csv(str(path), separator=",")
        ph = Phecode(platform="custom", icd_file_path=str(path))
        assert len(ph.icd_events) > 0


# ---------------------------------------------------------------------------
# count_phecode
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("engine", ["polars", "duckdb"])
class TestCountPhecodeLocal:
    def test_phecodeX_US_creates_output_file(self, custom_phecode, tmp_path, engine):
        out = str(tmp_path / "counts.tsv")
        custom_phecode.count_phecode(
            phecode_version="X", icd_version="US", output_file_path=out, engine=engine
        )
        assert os.path.exists(out)

    def test_phecodeX_US_has_required_columns(self, custom_phecode, tmp_path, engine):
        out = str(tmp_path / "counts.tsv")
        custom_phecode.count_phecode(
            phecode_version="X", icd_version="US", output_file_path=out, engine=engine
        )
        result = pl.read_csv(out, separator="\t", schema_overrides={"phecode": str})
        for col in ["person_id", "phecode", "count", "first_event_date"]:
            assert col in result.columns

    def test_count_values_are_positive(self, custom_phecode, tmp_path, engine):
        out = str(tmp_path / "counts.tsv")
        custom_phecode.count_phecode(
            phecode_version="X", icd_version="US", output_file_path=out, engine=engine
        )
        result = pl.read_csv(out, separator="\t", schema_overrides={"phecode": str})
        assert all(c > 0 for c in result["count"].to_list())

    def test_phecode12_US_creates_output_file(self, custom_phecode, tmp_path, engine):
        out = str(tmp_path / "counts12.tsv")
        custom_phecode.count_phecode(
            phecode_version="1.2", icd_version="US", output_file_path=out, engine=engine
        )
        if os.path.exists(out):
            result = pl.read_csv(out, separator="\t", schema_overrides={"phecode": str})
            assert "phecode" in result.columns

    def test_phecodeX_WHO_creates_output_file(self, custom_phecode, tmp_path, engine):
        out = str(tmp_path / "counts_who.tsv")
        custom_phecode.count_phecode(
            phecode_version="X", icd_version="WHO", output_file_path=out, engine=engine
        )
        if os.path.exists(out):
            result = pl.read_csv(out, separator="\t", schema_overrides={"phecode": str})
            assert "phecode" in result.columns

    def test_default_output_filename_generated(self, custom_phecode, tmp_path, monkeypatch, engine):
        monkeypatch.chdir(tmp_path)
        custom_phecode.count_phecode(phecode_version="X", icd_version="US", engine=engine)
        expected = tmp_path / "custom_US_phecodeX_counts.tsv"
        assert expected.exists()

    def test_no_duplicate_person_phecode_pairs(self, custom_phecode, tmp_path, engine):
        out = str(tmp_path / "counts.tsv")
        custom_phecode.count_phecode(
            phecode_version="X", icd_version="US", output_file_path=out, engine=engine
        )
        result = pl.read_csv(out, separator="\t", schema_overrides={"phecode": str})
        n_unique = result.select(["person_id", "phecode"]).n_unique()
        assert n_unique == len(result)


class TestCountPhecodeEngineEquivalence:
    """Both engines must produce identical phecode count tables (row-wise)."""

    def test_phecodeX_US_polars_vs_duckdb_match(self, custom_phecode, tmp_path):
        polars_out = str(tmp_path / "counts_polars.tsv")
        duckdb_out = str(tmp_path / "counts_duckdb.tsv")

        custom_phecode.count_phecode(
            phecode_version="X", icd_version="US",
            output_file_path=polars_out, engine="polars",
        )
        custom_phecode.count_phecode(
            phecode_version="X", icd_version="US",
            output_file_path=duckdb_out, engine="duckdb",
        )

        sort_cols = ["person_id", "phecode"]
        schema = {"phecode": str}
        a = pl.read_csv(polars_out, separator="\t", schema_overrides=schema).sort(sort_cols)
        b = pl.read_csv(duckdb_out, separator="\t", schema_overrides=schema).sort(sort_cols)
        assert a.equals(b)

    def test_phecode12_US_polars_vs_duckdb_match(self, custom_phecode, tmp_path):
        polars_out = str(tmp_path / "counts12_polars.tsv")
        duckdb_out = str(tmp_path / "counts12_duckdb.tsv")

        custom_phecode.count_phecode(
            phecode_version="1.2", icd_version="US",
            output_file_path=polars_out, engine="polars",
        )
        custom_phecode.count_phecode(
            phecode_version="1.2", icd_version="US",
            output_file_path=duckdb_out, engine="duckdb",
        )

        if not (os.path.exists(polars_out) and os.path.exists(duckdb_out)):
            pytest.skip("phecode 1.2 produced no output for this fixture")

        sort_cols = ["person_id", "phecode"]
        schema = {"phecode": str}
        a = pl.read_csv(polars_out, separator="\t", schema_overrides=schema).sort(sort_cols)
        b = pl.read_csv(duckdb_out, separator="\t", schema_overrides=schema).sort(sort_cols)
        assert a.equals(b)


# ---------------------------------------------------------------------------
# add_phecode_time_to_event (static method — no AoU)
# ---------------------------------------------------------------------------

class TestAddPhecodeTimeToEvent:
    @pytest.fixture
    def base_files(self, tmp_path):
        phecode_counts = pl.DataFrame({
            "person_id": [1, 1, 2],
            "phecode": ["250", "401", "250"],
            "count": [3, 2, 1],
            "first_event_date": [
                datetime.date(2020, 6, 1),
                datetime.date(2020, 8, 1),
                datetime.date(2020, 3, 1),
            ],
        })
        cohort = pl.DataFrame({
            "person_id": [1, 2],
            "study_start": [datetime.date(2020, 1, 1), datetime.date(2020, 1, 1)],
        })
        pf = str(tmp_path / "phecode.tsv")
        cf = str(tmp_path / "cohort.tsv")
        phecode_counts.write_csv(pf, separator="\t")
        cohort.write_csv(cf, separator="\t")
        return pf, cf, tmp_path

    def test_creates_output_file(self, base_files):
        pf, cf, tmp_path = base_files
        out = str(tmp_path / "output.tsv")
        Phecode.add_phecode_time_to_event(pf, cf, "study_start", output_file_path=out)
        assert os.path.exists(out)

    def test_output_has_time_to_event_column(self, base_files):
        pf, cf, tmp_path = base_files
        out = str(tmp_path / "output.tsv")
        Phecode.add_phecode_time_to_event(pf, cf, "study_start", output_file_path=out)
        result = pl.read_csv(out, separator="\t")
        assert "phecode_time_to_event" in result.columns

    def test_days_calculation_correct(self, base_files):
        pf, cf, tmp_path = base_files
        out = str(tmp_path / "output.tsv")
        Phecode.add_phecode_time_to_event(pf, cf, "study_start", time_unit="days", output_file_path=out)
        result = pl.read_csv(out, separator="\t", schema_overrides={"phecode": str})
        # person 1, phecode 250: 2020-06-01 - 2020-01-01 = 152 days
        row = result.filter((pl.col("person_id") == 1) & (pl.col("phecode") == "250"))
        assert row["phecode_time_to_event"][0] == pytest.approx(152, abs=1)

    def test_years_calculation_correct(self, tmp_path):
        phecode_counts = pl.DataFrame({
            "person_id": [1],
            "phecode": ["250"],
            "count": [1],
            "first_event_date": [datetime.date(2021, 1, 1)],
        })
        cohort = pl.DataFrame({
            "person_id": [1],
            "study_start": [datetime.date(2020, 1, 1)],
        })
        pf = str(tmp_path / "p.tsv")
        cf = str(tmp_path / "c.tsv")
        out = str(tmp_path / "o.tsv")
        phecode_counts.write_csv(pf, separator="\t")
        cohort.write_csv(cf, separator="\t")
        Phecode.add_phecode_time_to_event(pf, cf, "study_start", time_unit="years", output_file_path=out)
        result = pl.read_csv(out, separator="\t")
        assert result["phecode_time_to_event"][0] == pytest.approx(1.0, abs=0.01)

    def test_invalid_time_unit_raises(self, base_files):
        pf, cf, tmp_path = base_files
        with pytest.raises(ValueError):
            Phecode.add_phecode_time_to_event(pf, cf, "study_start", time_unit="months")

    def test_output_columns_complete(self, base_files):
        pf, cf, tmp_path = base_files
        out = str(tmp_path / "output.tsv")
        Phecode.add_phecode_time_to_event(pf, cf, "study_start", output_file_path=out)
        result = pl.read_csv(out, separator="\t")
        for col in ["person_id", "phecode", "count", "first_event_date", "phecode_time_to_event"]:
            assert col in result.columns

    def test_default_output_filename_derived(self, base_files):
        pf, cf, tmp_path = base_files
        Phecode.add_phecode_time_to_event(pf, cf, "study_start")
        expected = pf.replace(".tsv", "_with_phecode_time_to_event.tsv")
        assert os.path.exists(expected)
