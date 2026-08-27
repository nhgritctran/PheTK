"""
Unit tests for phecode.py — AoU BigQuery calls are mocked.
"""
import datetime
import pytest
import polars as pl
from unittest.mock import patch

from phetk.phecode import Phecode, available_phecode_versions, get_phecode_map


class TestGetPhecodeMap:
    def test_defaults_return_phecodeX_us(self):
        df = get_phecode_map()
        assert isinstance(df, pl.DataFrame)
        assert len(df) > 0
        for col in ["phecode", "ICD", "flag", "code_val"]:
            assert col in df.columns

    def test_alias_matches_pinned_version(self):
        assert get_phecode_map("X").equals(get_phecode_map("X1.0"))

    @pytest.mark.parametrize("value", ["x", "x1.0", " X "])
    def test_lowercase_and_whitespace_accepted(self, value):
        assert get_phecode_map(value).equals(get_phecode_map("X1.0"))

    def test_phecode12_has_exclusion_columns(self):
        df = get_phecode_map("1.2")
        assert "exclude_range" in df.columns
        assert "phecode_unrolled" in df.columns

    def test_who_mapping_loads(self):
        df = get_phecode_map("X", icd_version="WHO")
        assert len(df) > 0
        assert "phecode" in df.columns

    def test_core_columns_phecodeX(self):
        df = get_phecode_map("X", keep_all_columns=False)
        assert df.columns == ["phecode", "ICD", "flag"]

    def test_core_columns_phecode12(self):
        df = get_phecode_map("1.2", keep_all_columns=False)
        assert df.columns == ["phecode_unrolled", "ICD", "flag"]

    def test_flag_is_int8(self):
        assert get_phecode_map("X")["flag"].dtype == pl.Int8

    def test_custom_mapping_file(self, tmp_path):
        path = tmp_path / "custom_map.csv"
        path.write_text("phecode,ICD,flag,code_val\n001.1,E11,10,1.0\n002.2,I10,10,2.0\n")
        df = get_phecode_map("X", icd_version="custom", phecode_map_file_path=str(path))
        assert len(df) == 2
        assert df["phecode"].dtype == pl.Utf8

    def test_unsupported_version_raises(self):
        with pytest.raises(ValueError):
            get_phecode_map("2.0")

    def test_unsupported_version_icd_pair_raises(self):
        with pytest.raises(ValueError):
            get_phecode_map("1.2", icd_version="WHO")

    def test_icd_version_stays_strict(self):
        with pytest.raises(ValueError):
            get_phecode_map("X", icd_version="us")

    def test_custom_icd_without_path_raises(self):
        with pytest.raises(ValueError):
            get_phecode_map("X", icd_version="custom")

    def test_raises_rather_than_exits(self):
        with pytest.raises(ValueError):
            get_phecode_map("2.0")
        # SystemExit is not a subclass of ValueError, so confirm explicitly
        try:
            get_phecode_map("2.0")
        except SystemExit:
            pytest.fail("get_phecode_map must raise ValueError, not SystemExit")
        except ValueError:
            pass


class TestAvailablePhecodeVersions:
    def test_returns_canonical_versions(self):
        assert available_phecode_versions() == ["1.2", "X1.0"]

    def test_alias_not_included(self):
        assert "X" not in available_phecode_versions()

    def test_every_version_loads(self):
        for version in available_phecode_versions():
            assert len(get_phecode_map(version)) > 0


@pytest.fixture
def fake_icd_df():
    return pl.DataFrame({
        "person_id": [1, 2, 3],
        "date": [
            datetime.date(2020, 1, 1),
            datetime.date(2020, 6, 15),
            datetime.date(2021, 3, 1),
        ],
        "ICD": ["E11", "I10", "J45"],
        "vocabulary_id": ["ICD10CM", "ICD10CM", "ICD10CM"],
    })


@pytest.fixture
def aou_phecode(aou_env, fake_icd_df):
    with patch("phetk._utils.polars_gbq", return_value=fake_icd_df):
        return Phecode(platform="aou")


# ---------------------------------------------------------------------------
# Phecode.__init__ — AoU platform
# ---------------------------------------------------------------------------

class TestPhecodeInitAou:
    def test_cdr_set_from_env(self, aou_phecode):
        assert aou_phecode.cdr == "test_project.test_cdr"

    def test_icd_events_loaded(self, aou_phecode):
        assert aou_phecode.icd_events is not None
        assert len(aou_phecode.icd_events) > 0

    def test_flag_column_added(self, aou_phecode):
        assert "flag" in aou_phecode.icd_events.columns

    def test_icd10cm_gets_flag_10(self, aou_phecode):
        flags = aou_phecode.icd_events["flag"].to_list()
        assert all(f == 10 for f in flags)

    def test_icd9cm_gets_flag_9(self, aou_env):
        icd9_df = pl.DataFrame({
            "person_id": [1],
            "date": [datetime.date(2020, 1, 1)],
            "ICD": ["250.0"],
            "vocabulary_id": ["ICD9CM"],
        })
        with patch("phetk._utils.polars_gbq", return_value=icd9_df):
            ph = Phecode(platform="aou")
        assert ph.icd_events["flag"][0] == 9

    def test_gbq_dataset_id_overrides_env(self, aou_env, fake_icd_df):
        with patch("phetk._utils.polars_gbq", return_value=fake_icd_df):
            ph = Phecode(platform="aou", gbq_dataset_id="override.cdr")
        assert ph.cdr == "override.cdr"

    def test_existing_flag_column_preserved(self, aou_env):
        icd_with_flag = pl.DataFrame({
            "person_id": [1],
            "date": [datetime.date(2020, 1, 1)],
            "ICD": ["E11"],
            "vocabulary_id": ["ICD10CM"],
            "flag": pl.Series([10], dtype=pl.Int8),
        })
        with patch("phetk._utils.polars_gbq", return_value=icd_with_flag):
            ph = Phecode(platform="aou")
        assert "flag" in ph.icd_events.columns


# ---------------------------------------------------------------------------
# Phecode.__init__ — custom platform
# ---------------------------------------------------------------------------

class TestPhecodeInitCustom:
    def test_custom_without_file_exits(self):
        with pytest.raises(SystemExit):
            Phecode(platform="custom", icd_file_path=None)

    def test_invalid_platform_exits(self):
        with pytest.raises(SystemExit):
            Phecode(platform="invalid_platform")

    def test_custom_loads_file(self, icd_file):
        ph = Phecode(platform="custom", icd_file_path=icd_file)
        assert len(ph.icd_events) > 0

    def test_custom_adds_flag_column(self, icd_file):
        ph = Phecode(platform="custom", icd_file_path=icd_file)
        assert "flag" in ph.icd_events.columns


# ---------------------------------------------------------------------------
# Phecode.add_age_at_first_event (mocked BigQuery chunk)
# ---------------------------------------------------------------------------

class TestAddAgeAtFirstEvent:
    def test_age_at_first_event_column_created(self, aou_phecode, tmp_path):
        phecode_counts = pl.DataFrame({
            "person_id": [1, 2],
            "phecode": ["250", "401"],
            "count": [3, 2],
            "first_event_date": [datetime.date(2020, 6, 1), datetime.date(2021, 3, 1)],
        })
        phecode_file = str(tmp_path / "counts.tsv")
        phecode_counts.write_csv(phecode_file, separator="\t")
        out_file = str(tmp_path / "output.tsv")

        fake_dob = pl.DataFrame({
            "person_id": [1, 2],
            "date_of_birth": [datetime.date(1970, 1, 1), datetime.date(1985, 6, 15)],
            "year_of_birth": [1970, 1985],
            "current_age": [55.0, 40.0],
            "current_age_squared": [3025.0, 1600.0],
            "current_age_cubed": [166375.0, 64000.0],
        })

        with patch("phetk._utils.polars_gbq_chunk", return_value=fake_dob):
            aou_phecode.add_age_at_first_event(
                phecode_count_file_path=phecode_file,
                output_file_path=out_file,
            )

        result = pl.read_csv(out_file, separator="\t")
        assert "age_at_first_event" in result.columns

    def test_age_at_first_event_is_positive(self, aou_phecode, tmp_path):
        phecode_counts = pl.DataFrame({
            "person_id": [1],
            "phecode": ["250"],
            "count": [2],
            "first_event_date": [datetime.date(2020, 6, 1)],
        })
        phecode_file = str(tmp_path / "counts.tsv")
        phecode_counts.write_csv(phecode_file, separator="\t")
        out_file = str(tmp_path / "output.tsv")

        fake_dob = pl.DataFrame({
            "person_id": [1],
            "date_of_birth": [datetime.date(1970, 1, 1)],
            "year_of_birth": [1970],
            "current_age": [55.0],
            "current_age_squared": [3025.0],
            "current_age_cubed": [166375.0],
        })

        with patch("phetk._utils.polars_gbq_chunk", return_value=fake_dob):
            aou_phecode.add_age_at_first_event(
                phecode_count_file_path=phecode_file,
                output_file_path=out_file,
            )

        result = pl.read_csv(out_file, separator="\t")
        assert result["age_at_first_event"][0] > 0

    def test_output_has_expected_columns(self, aou_phecode, tmp_path):
        phecode_counts = pl.DataFrame({
            "person_id": [1],
            "phecode": ["250"],
            "count": [1],
            "first_event_date": [datetime.date(2020, 6, 1)],
        })
        phecode_file = str(tmp_path / "counts.tsv")
        phecode_counts.write_csv(phecode_file, separator="\t")
        out_file = str(tmp_path / "output.tsv")

        fake_dob = pl.DataFrame({
            "person_id": [1],
            "date_of_birth": [datetime.date(1970, 1, 1)],
            "year_of_birth": [1970],
            "current_age": [55.0],
            "current_age_squared": [3025.0],
            "current_age_cubed": [166375.0],
        })

        with patch("phetk._utils.polars_gbq_chunk", return_value=fake_dob):
            aou_phecode.add_age_at_first_event(
                phecode_count_file_path=phecode_file,
                output_file_path=out_file,
            )

        result = pl.read_csv(out_file, separator="\t")
        for col in ["person_id", "phecode", "count", "first_event_date", "age_at_first_event"]:
            assert col in result.columns
