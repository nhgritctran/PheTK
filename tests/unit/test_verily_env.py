"""
Unit tests for the Verily Workbench environment setup in _utils.py.

Covers bucket classification (`_is_user_bucket`, `_classify_wb_buckets`,
`_classify_gcloud_buckets`), the numbered `WORKSPACE_BUCKET_N` convention, and
`setup_verily_env` end to end with the `wb` CLI mocked. No subprocess is ever
actually spawned.

The workspace modelled here mirrors the shape of a real one — two controlled
user buckets, one referenced user bucket, and two AoU reference buckets — but
user bucket names are placeholders, since nothing under test depends on them.

The `vwb_aou_*` / `vwb-aou-*` names are kept verbatim: they are AoU platform
identifiers, and the separator disagreement between a resource id and the bucket
it points at (`vwb_aou_allxall_v8` → `vwb-aou-allxall`) is exactly what leaked an
AoU bucket into `WORKSPACE_REFERENCED_BUCKET2`.
"""
import json
import os
from unittest.mock import MagicMock, patch

import pytest

import phetk._utils as _utils
from phetk._utils import (
    _classify_gcloud_buckets,
    _classify_wb_buckets,
    _is_user_bucket,
    _print_env_summary,
    setup_verily_env,
)


@pytest.fixture(autouse=True)
def _isolate_env():
    """
    Snapshot and restore os.environ around every test.

    The functions under test assign to os.environ directly, which monkeypatch
    does not track, so a full snapshot/restore is required. Bucket-related
    variables are cleared up front so a developer's real Verily environment
    cannot influence assertions.
    """
    saved = dict(os.environ)
    for key in list(os.environ):
        if key.startswith(("WORKSPACE_", "GOOGLE_")):
            del os.environ[key]
    _utils._VERILY_WORKBENCH_CACHED = None
    yield
    os.environ.clear()
    os.environ.update(saved)
    _utils._VERILY_WORKBENCH_CACHED = None


# Shape of a `wb resource list` response, with placeholder user bucket names.
CONTROLLED_1 = ("user-bucket-a", "user-bucket-a", "CONTROLLED")
CONTROLLED_2 = ("user-bucket-b", "user-bucket-b", "CONTROLLED")
REFERENCED_USER = (
    "user-referenced-bucket",
    "user-referenced-bucket",
    "REFERENCED",
)
AOU_ALLXALL = ("vwb_aou_allxall_v8", "vwb-aou-allxall", "REFERENCED")
AOU_CONTROLLED = (
    "vwb-aou-datasets-controlled-v9",
    "vwb-aou-datasets-controlled",
    "REFERENCED",
)


# ---------------------------------------------------------------------------
# _is_user_bucket
# ---------------------------------------------------------------------------

class TestIsUserBucket:

    def test_aou_underscore_resource_id_hyphen_bucket_name(self):
        """The regression case: id uses underscores, bucket name uses hyphens."""
        assert _is_user_bucket("vwb_aou_allxall_v8", "vwb-aou-allxall") is False

    def test_aou_hyphen_form(self):
        assert _is_user_bucket(
            "vwb-aou-datasets-controlled-v9", "vwb-aou-datasets-controlled"
        ) is False

    def test_aou_underscores_in_both_fields(self):
        assert _is_user_bucket("vwb_aou_allxall_v8", "vwb_aou_allxall") is False

    def test_bucket_name_is_authoritative(self):
        """A resource id with no AoU marker at all is still caught by bucketName."""
        assert _is_user_bucket("allxall", "vwb-aou-allxall") is False

    def test_resource_id_catches_when_bucket_name_missing(self):
        assert _is_user_bucket("vwb_aou_allxall_v8", "") is False

    def test_case_insensitive(self):
        assert _is_user_bucket("VWB_AOU_AllxAll_v8", "VWB-AOU-ALLXALL") is False

    @pytest.mark.parametrize("name", [
        "dataproc-staging-us-central1-123-abc",
        "cloned-workspace-bucket",
    ])
    def test_infra_prefixes_on_bucket_name(self, name):
        assert _is_user_bucket("my-bucket", name) is False

    @pytest.mark.parametrize("rid", [
        "dataproc-staging-us-central1-123-abc",
        "cloned-workspace-bucket",
    ])
    def test_infra_prefixes_on_resource_id(self, rid):
        assert _is_user_bucket(rid, "my-bucket") is False

    @pytest.mark.parametrize("rid,bn", [
        ("user-bucket-a", "user-bucket-a"),
        ("user-referenced-bucket", "user-referenced-bucket"),
        ("", "some-user-bucket"),
        ("some-user-bucket", ""),
    ])
    def test_user_buckets_kept(self, rid, bn):
        assert _is_user_bucket(rid, bn) is True

    def test_aou_substring_not_at_start_is_kept(self):
        """Only a leading vwb-aou-/vwb_aou_ marks an AoU bucket."""
        assert _is_user_bucket("my-vwb-aou-copy", "my-vwb-aou-copy") is True


# ---------------------------------------------------------------------------
# _classify_wb_buckets
# ---------------------------------------------------------------------------

class TestClassifyWbBuckets:

    def test_observed_workspace(self, capsys):
        """
        Full ground-truth workspace: both AoU references are dropped, so the
        single remaining referenced bucket gets the unnumbered variable.
        """
        _classify_wb_buckets(
            [CONTROLLED_1, CONTROLLED_2, REFERENCED_USER, AOU_ALLXALL, AOU_CONTROLLED],
            [],
        )
        assert os.environ["WORKSPACE_BUCKET_1"] == "gs://user-bucket-a"
        assert os.environ["WORKSPACE_BUCKET_2"] == "gs://user-bucket-b"
        assert os.environ["WORKSPACE_REFERENCED_BUCKET"] == (
            "gs://user-referenced-bucket"
        )
        # The leak: no variable anywhere may point at the AoU bucket.
        assert not any("allxall" in v for v in os.environ.values())
        assert "WORKSPACE_REFERENCED_BUCKET_2" not in os.environ
        assert "Multiple referenced buckets" not in capsys.readouterr().out

    def test_aou_allxall_dropped(self):
        """Regression test for WORKSPACE_REFERENCED_BUCKET2=gs://vwb-aou-allxall."""
        _classify_wb_buckets([REFERENCED_USER, AOU_ALLXALL], [])
        assert os.environ["WORKSPACE_REFERENCED_BUCKET"] == (
            "gs://user-referenced-bucket"
        )
        assert "WORKSPACE_REFERENCED_BUCKET_1" not in os.environ

    def test_bucket_name_is_what_matters(self):
        """A benign resource id does not rescue an AoU bucket name."""
        _classify_wb_buckets([("allxall", "vwb-aou-allxall", "REFERENCED")], [])
        assert "WORKSPACE_REFERENCED_BUCKET" not in os.environ

    def test_single_controlled_bucket_is_unnumbered(self):
        _classify_wb_buckets([CONTROLLED_1], [])
        assert os.environ["WORKSPACE_BUCKET"] == "gs://user-bucket-a"
        assert "WORKSPACE_BUCKET_1" not in os.environ

    def test_multiple_controlled_buckets_numbered(self, capsys):
        _classify_wb_buckets([CONTROLLED_1, CONTROLLED_2], [])
        assert os.environ["WORKSPACE_BUCKET_1"] == "gs://user-bucket-a"
        assert os.environ["WORKSPACE_BUCKET_2"] == "gs://user-bucket-b"
        assert "WORKSPACE_BUCKET" not in os.environ
        out = capsys.readouterr().out
        assert "Multiple controlled buckets" in out
        assert "WORKSPACE_BUCKET_1" in out

    def test_multiple_referenced_buckets_numbered(self, capsys):
        _classify_wb_buckets(
            [REFERENCED_USER, ("other-ref", "other-ref-bucket", "REFERENCED")], []
        )
        assert os.environ["WORKSPACE_REFERENCED_BUCKET_1"] == (
            "gs://user-referenced-bucket"
        )
        assert os.environ["WORKSPACE_REFERENCED_BUCKET_2"] == "gs://other-ref-bucket"
        out = capsys.readouterr().out
        assert "Multiple referenced buckets" in out
        assert "WORKSPACE_REFERENCED_BUCKET_1" in out

    def test_infra_buckets_dropped(self):
        _classify_wb_buckets(
            [
                CONTROLLED_1,
                ("dataproc-staging-us-central1-1-x", "dataproc-staging-x", "CONTROLLED"),
                ("legit-id", "cloned-old-bucket", "CONTROLLED"),
            ],
            [],
        )
        assert os.environ["WORKSPACE_BUCKET"] == "gs://user-bucket-a"
        assert "WORKSPACE_BUCKET_1" not in os.environ

    def test_stewardship_split_preserved(self):
        _classify_wb_buckets([CONTROLLED_1, REFERENCED_USER], [])
        assert os.environ["WORKSPACE_BUCKET"] == "gs://user-bucket-a"
        assert os.environ["WORKSPACE_REFERENCED_BUCKET"] == (
            "gs://user-referenced-bucket"
        )

    def test_no_buckets_at_all_warns(self, capsys):
        _classify_wb_buckets([], [{"resourceType": "BQ_DATASET"}])
        out = capsys.readouterr().out
        assert "No GCS_BUCKET resources found" in out
        assert "BQ_DATASET" in out
        assert "WORKSPACE_BUCKET" not in os.environ

    def test_only_aou_buckets_warns(self, capsys):
        _classify_wb_buckets([AOU_ALLXALL, AOU_CONTROLLED], [])
        out = capsys.readouterr().out
        assert "No user-created GCS bucket found" in out
        assert "WORKSPACE_BUCKET" not in os.environ
        assert "WORKSPACE_REFERENCED_BUCKET" not in os.environ


# ---------------------------------------------------------------------------
# _classify_gcloud_buckets — must not drift from the wb path
# ---------------------------------------------------------------------------

class TestClassifyGcloudBuckets:

    def test_aou_and_infra_dropped(self):
        _classify_gcloud_buckets([
            "vwb-aou-allxall",
            "vwb_aou_allxall",
            "dataproc-staging-us-central1-1-x",
            "cloned-old-bucket",
            "user-bucket-a",
        ])
        assert os.environ["WORKSPACE_BUCKET"] == "gs://user-bucket-a"

    def test_multiple_numbered(self, capsys):
        _classify_gcloud_buckets(["user-bucket-a", "user-bucket-b"])
        assert os.environ["WORKSPACE_BUCKET_1"] == "gs://user-bucket-a"
        assert os.environ["WORKSPACE_BUCKET_2"] == "gs://user-bucket-b"
        out = capsys.readouterr().out
        assert "WORKSPACE_BUCKET_1=gs://user-bucket-a" in out
        assert "WORKSPACE_BUCKET_2=gs://user-bucket-b" in out

    def test_none_warns(self, capsys):
        _classify_gcloud_buckets(["vwb-aou-allxall"])
        assert "No user-created GCS bucket found" in capsys.readouterr().out
        assert "WORKSPACE_BUCKET" not in os.environ


# ---------------------------------------------------------------------------
# _print_env_summary — the probe must match the setters
# ---------------------------------------------------------------------------

class TestPrintEnvSummary:

    def test_lists_numbered_variables(self, capsys):
        _classify_wb_buckets([CONTROLLED_1, CONTROLLED_2, REFERENCED_USER], [])
        _print_env_summary(
            ["GOOGLE_PROJECT", "WORKSPACE_BUCKET", "WORKSPACE_REFERENCED_BUCKET"],
            set(),
        )
        out = capsys.readouterr().out
        assert "WORKSPACE_BUCKET_1=gs://user-bucket-a" in out
        assert "WORKSPACE_BUCKET_2=gs://user-bucket-b" in out
        assert "WORKSPACE_REFERENCED_BUCKET=gs://user-referenced-" in out

    def test_stops_at_first_gap(self, capsys):
        os.environ["WORKSPACE_BUCKET_1"] = "gs://one"
        os.environ["WORKSPACE_BUCKET_3"] = "gs://three"
        _print_env_summary(["WORKSPACE_BUCKET"], set())
        out = capsys.readouterr().out
        assert "WORKSPACE_BUCKET_1=gs://one" in out
        assert "gs://three" not in out

    def test_already_set_reported_separately(self, capsys):
        os.environ["GOOGLE_PROJECT"] = "test-project"
        _print_env_summary(["GOOGLE_PROJECT"], {"GOOGLE_PROJECT"})
        out = capsys.readouterr().out
        assert "Already set (skipped):" in out
        assert "GOOGLE_PROJECT=test-project" in out


# ---------------------------------------------------------------------------
# setup_verily_env — integration with the wb CLI mocked
# ---------------------------------------------------------------------------

def _result(stdout: str = "", stderr: str = "", returncode: int = 0):
    m = MagicMock()
    m.stdout = stdout
    m.stderr = stderr
    m.returncode = returncode
    return m


WB_RESOURCE_LIST = [
    {
        "resourceType": "BQ_DATASET",
        "id": "cdr-v9",
        "projectId": "cdr-project",
        "datasetId": "C2024Q3R5",
    },
    {
        "resourceType": "GCS_BUCKET",
        "id": "user-bucket-a",
        "bucketName": "user-bucket-a",
        "stewardshipType": "CONTROLLED",
    },
    {
        "resourceType": "GCS_BUCKET",
        "id": "user-bucket-b",
        "bucketName": "user-bucket-b",
        "stewardshipType": "CONTROLLED",
    },
    {
        "resourceType": "GCS_BUCKET",
        "id": "user-referenced-bucket",
        "bucketName": "user-referenced-bucket",
        "stewardshipType": "REFERENCED",
    },
    {
        "resourceType": "GCS_BUCKET",
        "id": "vwb-aou-datasets-controlled-v9",
        "bucketName": "vwb-aou-datasets-controlled",
        "stewardshipType": "REFERENCED",
    },
    {
        "resourceType": "GCS_BUCKET",
        "id": "vwb_aou_allxall_v8",
        "bucketName": "vwb-aou-allxall",
        "stewardshipType": "REFERENCED",
    },
    {
        "resourceType": "GCS_OBJECT",
        "id": "v9-genomics-folder",
        "bucketName": "vwb-aou-datasets-controlled",
        "stewardshipType": "REFERENCED",
    },
]


def _wb_router(cmd, *args, **kwargs):
    if isinstance(cmd, list):
        if cmd[:2] == ["wb", "workspace"]:
            return _result(stdout=json.dumps({"googleProjectId": "test-project-id"}))
        if cmd[:3] == ["wb", "resource", "list"]:
            return _result(stdout=json.dumps(WB_RESOURCE_LIST))
    return _result()


class TestSetupVerilyEnv:

    def test_observed_workspace_end_to_end(self, capsys):
        with patch("subprocess.run", side_effect=_wb_router):
            setup_verily_env()

        assert os.environ["GOOGLE_PROJECT"] == "test-project-id"
        assert os.environ["GOOGLE_CLOUD_PROJECT"] == "test-project-id"
        assert os.environ["WORKSPACE_CDR"] == (
            "cdr-project.C2024Q3R5"
        )
        assert os.environ["WORKSPACE_BUCKET_1"] == "gs://user-bucket-a"
        assert os.environ["WORKSPACE_BUCKET_2"] == "gs://user-bucket-b"
        assert os.environ["WORKSPACE_REFERENCED_BUCKET"] == (
            "gs://user-referenced-bucket"
        )
        assert not any("aou-allxall" in v for v in os.environ.values())

        out = capsys.readouterr().out
        assert "vwb-aou-allxall" not in out
        assert "Multiple referenced buckets" not in out
        assert "WORKSPACE_BUCKET_1=gs://user-bucket-a" in out
        assert "WORKSPACE_BUCKET_2=gs://user-bucket-b" in out

    def test_gcs_object_resource_ignored(self):
        with patch("subprocess.run", side_effect=_wb_router):
            setup_verily_env()
        assert "v9-genomics-folder" not in str(sorted(os.environ.items()))

    def test_no_wb_cli_is_silent(self, capsys):
        with patch("subprocess.run", side_effect=FileNotFoundError("wb not installed")):
            setup_verily_env()
        assert capsys.readouterr().out == ""
        assert "WORKSPACE_BUCKET" not in os.environ
