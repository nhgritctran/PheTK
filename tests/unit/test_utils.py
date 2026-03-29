"""
Unit tests for _utils.py — pure utility functions with no cloud dependencies.
"""
import argparse
import os
import pytest
import polars as pl
import pandas as pd

from phetk import _utils


class TestStrToBool:
    @pytest.mark.parametrize("value", ["yes", "true", "t", "y", "1", "True", "YES", "Y"])
    def test_truthy_strings(self, value):
        assert _utils.str_to_bool(value) is True

    @pytest.mark.parametrize("value", ["no", "false", "f", "n", "0", "False", "NO", "N"])
    def test_falsy_strings(self, value):
        assert _utils.str_to_bool(value) is False

    def test_bool_true_passthrough(self):
        assert _utils.str_to_bool(True) is True

    def test_bool_false_passthrough(self):
        assert _utils.str_to_bool(False) is False

    def test_invalid_string_raises(self):
        with pytest.raises(argparse.ArgumentTypeError):
            _utils.str_to_bool("maybe")

    def test_empty_string_raises(self):
        with pytest.raises(argparse.ArgumentTypeError):
            _utils.str_to_bool("")


class TestToPolars:
    def test_polars_df_returned_unchanged(self):
        df = pl.DataFrame({"a": [1, 2, 3]})
        result = _utils.to_polars(df)
        assert isinstance(result, pl.DataFrame)
        assert result.equals(df)

    def test_pandas_df_converted(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        result = _utils.to_polars(df)
        assert isinstance(result, pl.DataFrame)
        assert result["a"].to_list() == [1, 2, 3]
        assert result["b"].to_list() == ["x", "y", "z"]


class TestHasOverlappingValues:
    def test_no_overlap_string_values(self):
        d = {0: "0/0", 1: "0/1", 2: "1/1"}
        assert _utils.has_overlapping_values(d) is False

    def test_no_overlap_list_values(self):
        d = {0: "0/0", 1: ["0/1", "1/1"]}
        assert _utils.has_overlapping_values(d) is False

    def test_overlap_within_lists(self):
        d = {0: ["0/0", "0/1"], 1: ["0/1", "1/1"]}
        assert _utils.has_overlapping_values(d) is True

    def test_overlap_string_and_list(self):
        d = {0: "0/0", 1: ["0/0", "1/1"]}
        assert _utils.has_overlapping_values(d) is True

    def test_single_entry_no_overlap(self):
        d = {0: "0/0"}
        assert _utils.has_overlapping_values(d) is False

    def test_empty_dict(self):
        assert _utils.has_overlapping_values({}) is False


class TestDetectDelimiter:
    def test_tsv_file(self, tmp_path):
        path = tmp_path / "test.tsv"
        path.write_text("col1\tcol2\tcol3\n1\t2\t3\n")
        assert _utils.detect_delimiter(str(path)) == "\t"

    def test_csv_file(self, tmp_path):
        path = tmp_path / "test.csv"
        path.write_text("col1,col2,col3\n1,2,3\n")
        assert _utils.detect_delimiter(str(path)) == ","

    def test_nonexistent_file_exits(self, tmp_path):
        with pytest.raises(SystemExit):
            _utils.detect_delimiter(str(tmp_path / "nonexistent.tsv"))


class TestGetPhecodeMapping:
    def test_phecodeX_US_returns_dataframe(self):
        df = _utils.get_phecode_mapping_table("X", "US", None, keep_all_columns=False)
        assert isinstance(df, pl.DataFrame)

    def test_phecodeX_US_has_required_columns(self):
        df = _utils.get_phecode_mapping_table("X", "US", None, keep_all_columns=False)
        assert "phecode" in df.columns
        assert "ICD" in df.columns
        assert "flag" in df.columns

    def test_phecodeX_WHO_returns_dataframe(self):
        df = _utils.get_phecode_mapping_table("X", "WHO", None, keep_all_columns=False)
        assert isinstance(df, pl.DataFrame)
        assert "phecode" in df.columns

    def test_phecode12_US_returns_dataframe(self):
        df = _utils.get_phecode_mapping_table("1.2", "US", None, keep_all_columns=False)
        assert isinstance(df, pl.DataFrame)
        assert "phecode_unrolled" in df.columns

    def test_phecodeX_US_keep_all_columns(self):
        df = _utils.get_phecode_mapping_table("X", "US", None, keep_all_columns=True)
        assert "code_val" in df.columns

    def test_invalid_phecode_version_exits(self):
        with pytest.raises(SystemExit):
            _utils.get_phecode_mapping_table("2.0", "US", None)

    def test_phecode12_WHO_exits(self):
        with pytest.raises(SystemExit):
            _utils.get_phecode_mapping_table("1.2", "WHO", None)

    def test_custom_icd_without_path_exits(self):
        with pytest.raises(SystemExit):
            _utils.get_phecode_mapping_table("X", "custom", None)

    def test_invalid_icd_version_exits(self):
        with pytest.raises(SystemExit):
            _utils.get_phecode_mapping_table("X", "unknown_version", None)

    def test_custom_mapping_file(self, tmp_path):
        path = tmp_path / "custom_map.csv"
        path.write_text("phecode,ICD,flag,code_val\n001.1,E11,10,1.0\n002.2,I10,10,2.0\n")
        df = _utils.get_phecode_mapping_table("X", "custom", str(path), keep_all_columns=False)
        assert "phecode" in df.columns
        assert len(df) == 2

    def test_flag_column_is_int8(self):
        df = _utils.get_phecode_mapping_table("X", "US", None, keep_all_columns=False)
        assert df["flag"].dtype == pl.Int8

    def test_not_empty(self):
        df = _utils.get_phecode_mapping_table("X", "US", None)
        assert len(df) > 0


class TestSaveLoadPickle:
    def test_roundtrip_dict(self, tmp_path):
        path = str(tmp_path / "test.pkl")
        obj = {"key": [1, 2, 3], "nested": {"a": 1}}
        _utils.save_pickle_object(obj, path)
        loaded = _utils.load_pickle_object(path)
        assert loaded == obj

    def test_roundtrip_list(self, tmp_path):
        path = str(tmp_path / "list.pkl")
        obj = [1, "two", 3.0, None]
        _utils.save_pickle_object(obj, path)
        assert _utils.load_pickle_object(path) == obj

    def test_file_created(self, tmp_path):
        path = str(tmp_path / "obj.pkl")
        _utils.save_pickle_object({"x": 1}, path)
        assert os.path.exists(path)


class TestGenerateChunkQueries:
    def _dummy_query(self, ds, ids):
        return f"SELECT * FROM {ds} WHERE id IN {ids}"

    def test_correct_number_of_chunks(self):
        queries = _utils.generate_chunk_queries(self._dummy_query, "ds", list(range(2500)), chunk_size=1000)
        assert len(queries) == 3

    def test_single_chunk(self):
        queries = _utils.generate_chunk_queries(self._dummy_query, "ds", list(range(50)), chunk_size=1000)
        assert len(queries) >= 1

    def test_query_contains_dataset(self):
        queries = _utils.generate_chunk_queries(self._dummy_query, "myds", list(range(10)), chunk_size=100)
        assert all("myds" in q for q in queries)


class TestSampleTsvFile:
    def test_creates_sample_file(self, tmp_path):
        path = tmp_path / "input.tsv"
        path.write_text("col1\tcol2\n" + "\n".join(f"{i}\t{i * 2}" for i in range(100)))
        _utils.sample_tsv_file(str(path), sample_ratio=0.1)
        sample_path = tmp_path / "input_sample_10.0pct.tsv"
        assert sample_path.exists()

    def test_sample_is_smaller(self, tmp_path):
        path = tmp_path / "input.tsv"
        rows = "\n".join(f"{i}\t{i}" for i in range(200))
        path.write_text("col1\tcol2\n" + rows)
        _utils.sample_tsv_file(str(path), sample_ratio=0.1)
        sample_path = tmp_path / "input_sample_10.0pct.tsv"
        original = pl.read_csv(str(path), separator="\t")
        sampled = pl.read_csv(str(sample_path), separator="\t")
        assert len(sampled) < len(original)

    def test_invalid_ratio_raises(self, tmp_path):
        path = tmp_path / "input.tsv"
        path.write_text("col1\n1\n")
        with pytest.raises(ValueError):
            _utils.sample_tsv_file(str(path), sample_ratio=1.5)

    def test_zero_ratio_raises(self, tmp_path):
        path = tmp_path / "input.tsv"
        path.write_text("col1\n1\n")
        with pytest.raises(ValueError):
            _utils.sample_tsv_file(str(path), sample_ratio=0.0)
