"""
Unit tests for _utils.py — pure utility functions with no cloud dependencies.
"""
import argparse
import os
import re
import subprocess
import pytest
import polars as pl
import pandas as pd
from unittest.mock import patch, MagicMock, call

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


class TestPhecodeMapRegistry:
    def test_every_registered_file_exists(self):
        phecode_dir = os.path.join(os.path.dirname(_utils.__file__), "phecode")
        for key, filename in _utils.PHECODE_MAP_FILES.items():
            assert os.path.exists(os.path.join(phecode_dir, filename)), key

    def test_every_registered_version_has_schema(self):
        for version, _ in _utils.PHECODE_MAP_FILES:
            assert version in _utils.PHECODE_MAP_SCHEMAS

    def test_every_schema_version_has_a_file(self):
        versions_with_files = {version for version, _ in _utils.PHECODE_MAP_FILES}
        assert set(_utils.PHECODE_MAP_SCHEMAS) == versions_with_files

    def test_alias_targets_are_canonical(self):
        for target in _utils.PHECODE_VERSION_ALIASES.values():
            assert target in _utils.PHECODE_MAP_SCHEMAS

    def test_x_alias_points_at_latest(self):
        assert _utils.resolve_phecode_version("X") == _utils.LATEST_PHECODE_X_VERSION

    @pytest.mark.parametrize("value", ["X", "x", " X ", "X1.0", "x1.0", " x1.0 "])
    def test_phecodeX_spellings_resolve(self, value):
        assert _utils.resolve_phecode_version(value) == "X1.0"

    @pytest.mark.parametrize("value", ["1.2", " 1.2 "])
    def test_phecode12_spellings_resolve(self, value):
        assert _utils.resolve_phecode_version(value) == "1.2"

    def test_unsupported_version_raises_listing_versions(self):
        with pytest.raises(ValueError) as excinfo:
            _utils.resolve_phecode_version("2.0")
        message = str(excinfo.value)
        assert "2.0" in message
        assert "1.2" in message
        assert "X1.0" in message

    def test_non_string_version_raises_value_error(self):
        with pytest.raises(ValueError):
            _utils.resolve_phecode_version(1.2)

    @pytest.mark.parametrize("value,family", [("X", "X"), ("X1.0", "X"), ("1.2", "1.2")])
    def test_phecode_version_family(self, value, family):
        assert _utils.phecode_version_family(value) == family

    def test_phecode_version_family_invalid_raises(self):
        with pytest.raises(ValueError):
            _utils.phecode_version_family("2.0")

    def test_available_phecode_versions_sorted(self):
        versions = _utils.available_phecode_versions()
        assert versions == ["1.2", "X1.0"]
        assert versions == sorted(versions)

    def test_accepted_phecode_versions_includes_aliases(self):
        accepted = _utils.accepted_phecode_versions()
        assert accepted == ["1.2", "X", "X1.0"]
        assert accepted == sorted(accepted)

    @pytest.mark.parametrize("value,icds", [("X", ["US", "WHO"]), ("1.2", ["US"])])
    def test_available_icd_versions(self, value, icds):
        assert _utils.available_icd_versions(value) == icds


class TestLoadPhecodeMap:
    def test_alias_and_pin_return_identical_frames(self):
        assert _utils.load_phecode_map("X", "US").equals(_utils.load_phecode_map("X1.0", "US"))

    def test_unsupported_version_raises_value_error(self):
        with pytest.raises(ValueError):
            _utils.load_phecode_map("2.0", "US")

    def test_unsupported_version_icd_pair_raises_value_error(self):
        with pytest.raises(ValueError):
            _utils.load_phecode_map("1.2", "WHO")

    def test_invalid_icd_version_raises_value_error(self):
        with pytest.raises(ValueError):
            _utils.load_phecode_map("X", "unknown_version")

    def test_custom_icd_without_path_raises_value_error(self):
        with pytest.raises(ValueError):
            _utils.load_phecode_map("X", "custom")

    @pytest.mark.parametrize("args", [
        ("2.0", "US"), ("1.2", "WHO"), ("X", "unknown_version"), ("X", "custom"),
    ])
    def test_errors_are_not_system_exit(self, args):
        with pytest.raises(ValueError):
            _utils.load_phecode_map(*args)

    def test_legacy_wrapper_accepts_pinned_version(self):
        df = _utils.get_phecode_mapping_table("X1.0", "US", None, keep_all_columns=False)
        assert df.columns == ["phecode", "ICD", "flag"]

    def test_no_deprecation_warning(self, recwarn):
        _utils.load_phecode_map("X", "US")
        _utils.load_phecode_map("1.2", "US")
        assert [w for w in recwarn if issubclass(w.category, DeprecationWarning)] == []


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


class TestGcsfsWrite:
    """Tests for the gcsfs_write() helper."""

    def _make_df(self):
        return pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})

    def _mock_gcsfs(self):
        """Create a mock gcsfs module and return (module_mock, file_mock)."""
        mock_mod = MagicMock()
        mock_fs = MagicMock()
        mock_mod.GCSFileSystem.return_value = mock_fs
        mock_file = MagicMock()
        mock_fs.open.return_value.__enter__ = MagicMock(return_value=mock_file)
        mock_fs.open.return_value.__exit__ = MagicMock(return_value=False)
        return mock_mod, mock_file

    def test_tsv_calls_write_csv_with_tab(self):
        mock_mod, mock_file = self._mock_gcsfs()
        df = self._make_df()
        with patch.dict("sys.modules", {"gcsfs": mock_mod}):
            with patch.object(df, "write_csv") as mock_write:
                _utils.gcsfs_write(df, "gs://bucket/output.tsv")
                mock_write.assert_called_once_with(mock_file, separator="\t")

    def test_csv_calls_write_csv(self):
        mock_mod, mock_file = self._mock_gcsfs()
        df = self._make_df()
        with patch.dict("sys.modules", {"gcsfs": mock_mod}):
            with patch.object(df, "write_csv") as mock_write:
                _utils.gcsfs_write(df, "gs://bucket/output.csv")
                mock_write.assert_called_once_with(mock_file)

    def test_parquet_calls_write_parquet(self):
        mock_mod, mock_file = self._mock_gcsfs()
        df = self._make_df()
        with patch.dict("sys.modules", {"gcsfs": mock_mod}):
            with patch.object(df, "write_parquet") as mock_write:
                _utils.gcsfs_write(df, "gs://bucket/output.parquet")
                mock_write.assert_called_once_with(mock_file)

    def test_unsupported_extension_raises(self):
        df = self._make_df()
        with pytest.raises(ValueError, match="Unsupported format"):
            _utils.gcsfs_write(df, "gs://bucket/output.xlsx")

    def test_file_format_override(self):
        mock_mod, mock_file = self._mock_gcsfs()
        df = self._make_df()
        with patch.dict("sys.modules", {"gcsfs": mock_mod}):
            with patch.object(df, "write_csv") as mock_write:
                # Extension is .dat (unsupported), but override says tsv
                _utils.gcsfs_write(df, "gs://bucket/output.dat", file_format="tsv")
                mock_write.assert_called_once_with(mock_file, separator="\t")


class TestWriteTsvFallback:
    """Tests for write_tsv() local writes and GCS 3-tier fallback."""

    def _make_df(self):
        return pl.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})

    def test_local_write(self, tmp_path):
        df = self._make_df()
        out = str(tmp_path / "out.tsv")
        _utils.write_tsv(df, out)
        result = pl.read_csv(out, separator="\t")
        assert result.shape == (3, 2)
        assert result["col1"].to_list() == [1, 2, 3]

    @patch("phetk._utils._to_gs_uri_if_bucket_mount", return_value="gs://b/k.tsv")
    def test_tier1_success(self, mock_mount):
        df = self._make_df()
        with patch.object(df, "write_csv") as mock_write:
            _utils.write_tsv(df, "gs://b/k.tsv")
            mock_write.assert_called_once_with("gs://b/k.tsv", separator="\t")

    @patch("phetk._utils._to_gs_uri_if_bucket_mount", return_value="gs://b/k.tsv")
    @patch("phetk._utils.gcsfs_write")
    def test_tier1_fails_tier2_succeeds(self, mock_gcsfs_write, mock_mount):
        df = self._make_df()
        original_write_csv = df.write_csv

        def side_effect(path, **kwargs):
            if isinstance(path, str) and path.startswith("gs://"):
                raise OSError("no object-store backend")
            return original_write_csv(path, **kwargs)

        with patch.object(df, "write_csv", side_effect=side_effect):
            _utils.write_tsv(df, "gs://b/k.tsv")
        mock_gcsfs_write.assert_called_once_with(df, "gs://b/k.tsv", file_format="tsv")

    @patch("phetk._utils._to_gs_uri_if_bucket_mount", return_value="gs://b/k.tsv")
    @patch("phetk._utils.gcsfs_write", side_effect=Exception("gcsfs fail"))
    @patch("shutil.which", return_value="/usr/bin/gcloud")
    @patch("subprocess.run")
    def test_tier1_tier2_fail_tier3_succeeds(
        self, mock_run, mock_which, mock_gcsfs_write, mock_mount
    ):
        mock_run.return_value = MagicMock(returncode=0)
        df = self._make_df()
        original_write_csv = df.write_csv

        def side_effect(path, **kwargs):
            if isinstance(path, str) and path.startswith("gs://"):
                raise OSError("no object-store backend")
            return original_write_csv(path, **kwargs)

        with patch.object(df, "write_csv", side_effect=side_effect):
            _utils.write_tsv(df, "gs://b/k.tsv")
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert args[:3] == ["gcloud", "storage", "cp"]
        assert args[4] == "gs://b/k.tsv"

    @patch("phetk._utils._to_gs_uri_if_bucket_mount", return_value="gs://b/k.tsv")
    @patch("phetk._utils.gcsfs_write", side_effect=Exception("gcsfs fail"))
    @patch("shutil.which", return_value="/usr/bin/gcloud")
    @patch("subprocess.run")
    def test_all_tiers_fail(
        self, mock_run, mock_which, mock_gcsfs_write, mock_mount
    ):
        mock_run.return_value = MagicMock(returncode=1, stderr="upload error")
        df = self._make_df()
        original_write_csv = df.write_csv

        def side_effect(path, **kwargs):
            if isinstance(path, str) and path.startswith("gs://"):
                raise OSError("no object-store backend")
            return original_write_csv(path, **kwargs)

        with patch.object(df, "write_csv", side_effect=side_effect):
            with pytest.raises(RuntimeError, match="gcloud storage cp failed"):
                _utils.write_tsv(df, "gs://b/k.tsv")


# Stand-in for the multi-line, XML-bearing message a cloud storage backend
# returns on a permission error. Values are placeholders on purpose: the point
# is that none of it reaches the user, so the specific text is irrelevant and
# no real bucket, project or service account belongs in the repo.
NOISY_CLOUD_ERROR = (
    "object-store error: The operation lacked the necessary privileges to\n"
    "complete for path some/output.tsv: Error performing POST\n"
    "https://storage.googleapis.com/example%2Dbucket/some%2Foutput%2Etsv?uploads=\n"
    "Server returned non-2xx status code: 403 Forbidden: <?xml version='1.0'?>\n"
    "<Error><Code>AccessDenied</Code><Details>service-account@example.iam."
    "gserviceaccount.com does not have storage.multipartUploads.create access"
    "</Details></Error>"
)


class TestWriteTsvOutput:
    """
    Tests the reporting of the GCS fallback, not the writing itself (which is
    covered by TestWriteTsvFallback).

    A write that succeeds via a fallback must not read like a failure. On
    AoU/Verily a service account can commonly stream but cannot do a native
    multipart write, so the first method always fails and the second always
    succeeds. The old output dumped the raw cloud exception between "Writing
    output to ..." and "Successfully generated ...", which read as a hard
    error even though the file was written.
    """

    def _make_df(self):
        return pl.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})

    def _direct_write_fails(self, df, exc):
        original = df.write_csv

        def side_effect(path, **kwargs):
            if isinstance(path, str) and path.startswith("gs://"):
                raise exc
            return original(path, **kwargs)

        return side_effect

    @patch("phetk._utils._to_gs_uri_if_bucket_mount", return_value="gs://b/k.tsv")
    def test_silent_when_first_method_works(self, mock_mount, capsys):
        df = self._make_df()
        with patch.object(df, "write_csv"):
            _utils.write_tsv(df, "gs://b/k.tsv")
        assert capsys.readouterr().out == ""

    @patch("phetk._utils._to_gs_uri_if_bucket_mount", return_value="gs://b/k.tsv")
    @patch("phetk._utils.gcsfs_write")
    def test_notice_reports_type_only(self, mock_gcsfs, mock_mount, capsys):
        """
        Two short lines, naming the exception type and nothing from its message.
        """
        df = self._make_df()
        with patch.object(
            df, "write_csv",
            side_effect=self._direct_write_fails(df, OSError(NOISY_CLOUD_ERROR)),
        ):
            _utils.write_tsv(df, "gs://b/k.tsv")

        out = capsys.readouterr().out
        lines = [ln for ln in out.splitlines() if ln.strip()]
        assert len(lines) == 2
        assert lines[0].strip() == (
            "Note: direct write unavailable (OSError); "
            "falling back to streaming write (gcsfs)..."
        )
        assert lines[1].strip() == "Saved using streaming write (gcsfs)."

    @patch("phetk._utils._to_gs_uri_if_bucket_mount", return_value="gs://b/k.tsv")
    @patch("phetk._utils.gcsfs_write")
    def test_no_part_of_the_message_is_printed(
        self, mock_gcsfs, mock_mount, capsys
    ):
        """
        Generic: every word of the exception message is withheld, whatever it
        says. Guards against reintroducing truncation, which would still leak
        paths, URLs and service account names into the notice.

        Parenthesised spans are removed before scanning: they hold the
        exception type and the method labels, which are PheTK's own text.
        test_notice_reports_type_only pins their exact content.
        """
        df = self._make_df()
        with patch.object(
            df, "write_csv",
            side_effect=self._direct_write_fails(df, OSError(NOISY_CLOUD_ERROR)),
        ):
            _utils.write_tsv(df, "gs://b/k.tsv")

        prose = re.sub(r"\([^)]*\)", "", capsys.readouterr().out)
        for token in set(re.findall(r"[A-Za-z0-9_.%/@-]{4,}", NOISY_CLOUD_ERROR)):
            assert token not in prose, f"leaked {token!r} from the exception message"

    @patch("phetk._utils._to_gs_uri_if_bucket_mount", return_value="gs://b/k.tsv")
    @patch("phetk._utils.gcsfs_write")
    def test_no_failure_wording_when_write_succeeds(
        self, mock_gcsfs, mock_mount, capsys
    ):
        df = self._make_df()
        with patch.object(
            df, "write_csv",
            side_effect=self._direct_write_fails(df, OSError(NOISY_CLOUD_ERROR)),
        ):
            _utils.write_tsv(df, "gs://b/k.tsv")

        out = capsys.readouterr().out
        for word in ("failed", "Error:", "Warning", "Traceback", "tier"):
            assert word.lower() not in out.lower()
        assert out.lstrip().startswith("Note:")

    @patch("phetk._utils._to_gs_uri_if_bucket_mount", return_value="gs://b/k.tsv")
    @patch("phetk._utils.gcsfs_write", side_effect=Exception("gcsfs fail"))
    @patch("shutil.which", return_value="/usr/bin/gcloud")
    @patch("subprocess.run")
    def test_second_fallback_names_third_method(
        self, mock_run, mock_which, mock_gcsfs, mock_mount, capsys
    ):
        mock_run.return_value = MagicMock(returncode=0)
        df = self._make_df()
        with patch.object(
            df, "write_csv",
            side_effect=self._direct_write_fails(df, OSError("nope")),
        ):
            _utils.write_tsv(df, "gs://b/k.tsv")

        out = capsys.readouterr().out
        assert "falling back to streaming write (gcsfs)" in out
        assert "falling back to local staging + gcloud" in out
        assert "Saved using local staging + gcloud." in out

    @patch("phetk._utils._to_gs_uri_if_bucket_mount", return_value="gs://b/k.tsv")
    @patch("phetk._utils.gcsfs_write", side_effect=Exception("gcsfs fail"))
    @patch("shutil.which", return_value="/usr/bin/gcloud")
    @patch("subprocess.run")
    def test_all_fail_reports_every_method_in_full(
        self, mock_run, mock_which, mock_gcsfs, mock_mount, capsys
    ):
        """Messages are withheld while retrying, but never lost."""
        mock_run.return_value = MagicMock(returncode=1, stderr="upload error")
        df = self._make_df()
        with patch.object(
            df, "write_csv",
            side_effect=self._direct_write_fails(df, OSError(NOISY_CLOUD_ERROR)),
        ):
            with pytest.raises(RuntimeError) as excinfo:
                _utils.write_tsv(df, "gs://b/k.tsv")

        msg = str(excinfo.value)
        assert "gs://b/k.tsv" in msg
        assert "direct write" in msg
        assert "streaming write (gcsfs): gcsfs fail" in msg
        assert "gcloud storage cp failed" in msg and "upload error" in msg
        # Full message, not just the type — this is the debugging path.
        assert "AccessDenied" in msg
        assert excinfo.value.__cause__ is not None
        # No "Saved using ..." claim when nothing was saved.
        assert "Saved using" not in capsys.readouterr().out

    @patch("phetk._utils._to_gs_uri_if_bucket_mount", return_value="gs://b/k.tsv")
    @patch("phetk._utils.gcsfs_write", side_effect=Exception("gcsfs fail"))
    @patch("shutil.which", return_value=None)
    def test_missing_gcloud_reported_in_summary(
        self, mock_which, mock_gcsfs, mock_mount
    ):
        df = self._make_df()
        with patch.object(
            df, "write_csv",
            side_effect=self._direct_write_fails(df, OSError("nope")),
        ):
            with pytest.raises(RuntimeError, match="requires the 'gcloud' CLI"):
                _utils.write_tsv(df, "gs://b/k.tsv")
