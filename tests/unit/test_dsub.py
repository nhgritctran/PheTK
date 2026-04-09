"""
Unit tests for _dsub.py — test Dsub construction, command generation, and argument merging.
No actual GCP/dsub calls are made.
"""
import subprocess
from unittest.mock import patch, MagicMock

import pytest

import phetk._utils as _utils
from phetk._dsub import Dsub, _check_dsub_version, _PINNED_DSUB_VERSION


@pytest.fixture(autouse=True)
def _reset_verily_cache():
    """
    Reset the module-level Verily detection cache before and after every test
    so cross-test contamination is impossible.
    """
    _utils._VERILY_WORKBENCH_CACHED = None
    yield
    _utils._VERILY_WORKBENCH_CACHED = None


def _make_subprocess_result(stdout: str = "", stderr: str = "", returncode: int = 0):
    """Build a MagicMock that mimics a subprocess.CompletedProcess."""
    m = MagicMock()
    m.stdout = stdout
    m.stderr = stderr
    m.returncode = returncode
    return m


def _default_subprocess_router(cmd, *args, **kwargs):
    """
    Default subprocess.run stub used by the autouse fixture.

    Routes by inspecting the command list:
        - `wb workspace ...` → FileNotFoundError (i.e., not on Verily)
        - `dsub --version`   → returncode=0, stdout="dsub version 0.5.0"
        - anything else      → harmless empty-stdout success result
    """
    if isinstance(cmd, list):
        if cmd[:2] == ["wb", "workspace"]:
            raise FileNotFoundError("wb not installed")
        if cmd[:1] == ["dsub"] and "--version" in cmd:
            return _make_subprocess_result(
                stdout=f"dsub version {_PINNED_DSUB_VERSION}\n",
            )
    return _make_subprocess_result()


@pytest.fixture(autouse=True)
def _mock_dsub_and_wb_subprocess(request):
    """
    Globally stub out subprocess.run for the external CLIs this module probes
    (`dsub --version` and `wb workspace describe`) so no test ever shells out.

    Tests that need to customise subprocess behaviour should declare the
    explicit `mock_subprocess` fixture, in which case this fixture yields
    without patching.
    """
    if "mock_subprocess" in request.fixturenames:
        yield
        return

    with patch("subprocess.run", side_effect=_default_subprocess_router):
        yield


@pytest.fixture
def mock_subprocess():
    """
    Single subprocess.run patch shared by `phetk._dsub` and `phetk._utils`.

    Both modules `import subprocess` at module level, so a single patch of
    `subprocess.run` replaces the attribute globally for both. Tests that
    exercise only one CLI can use `.return_value`; tests that trigger both
    (e.g. `Dsub.__init__`, which calls both `is_verily_workbench` and
    `_check_dsub_version`) should use `.side_effect` with a routing function.

    The returned dict aliases both `"dsub"` and `"utils"` to the same mock
    so existing dict-style access patterns continue to work.
    """
    with patch("subprocess.run") as m:
        yield {"dsub": m, "utils": m, "run": m}


def _routing_side_effect(wb_returncode: int = 0, dsub_version: str = _PINNED_DSUB_VERSION):
    """
    Build a side_effect callable that routes `subprocess.run` calls by the
    command keyword. Used in tests where Dsub.__init__ triggers both a
    `wb workspace describe` and a `dsub --version` subprocess call.
    """
    def _fake(cmd, *args, **kwargs):
        if isinstance(cmd, list):
            if cmd[:2] == ["wb", "workspace"]:
                return _make_subprocess_result(
                    stdout="{}", returncode=wb_returncode,
                )
            if cmd[:1] == ["dsub"] and "--version" in cmd:
                return _make_subprocess_result(
                    stdout=f"dsub version {dsub_version}\n",
                )
        return _make_subprocess_result()
    return _fake


@pytest.fixture
def dsub(aou_env):
    return Dsub(docker_image="phetk/phetk:latest", use_aou_docker_prefix=False)


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestDsubInit:
    def test_default_machine_type(self, dsub):
        assert dsub.machine_type == "c2d-highcpu-4"

    def test_default_provider(self, dsub):
        assert dsub.provider == "google-batch"

    def test_default_region(self, dsub):
        assert dsub.region == "us-central1"

    def test_default_boot_disk_size(self, dsub):
        assert dsub.boot_disk_size == 50

    def test_default_disk_size(self, dsub):
        assert dsub.disk_size == 256

    def test_job_name_auto_generated(self, dsub):
        assert dsub.job_name.startswith("phetk-")

    def test_job_name_lowercase(self, aou_env):
        d = Dsub(docker_image="img", job_name="MyJob", use_aou_docker_prefix=False)
        assert d.job_name == d.job_name.lower()

    def test_job_name_spaces_replaced_with_dashes(self, aou_env):
        d = Dsub(docker_image="img", job_name="my job", use_aou_docker_prefix=False)
        assert " " not in d.job_name
        assert "-" in d.job_name

    def test_job_name_underscores_replaced_with_dashes(self, aou_env):
        d = Dsub(docker_image="img", job_name="my_job", use_aou_docker_prefix=False)
        assert "_" not in d.job_name

    def test_job_name_starting_with_digit_prefixed(self, aou_env):
        d = Dsub(docker_image="img", job_name="123job", use_aou_docker_prefix=False)
        assert d.job_name[0].isalpha()

    @pytest.mark.aou
    def test_bucket_from_env(self, aou_env):
        d = Dsub(docker_image="img", use_aou_docker_prefix=False)
        assert d.bucket == "gs://test-bucket"

    @pytest.mark.aou
    def test_project_from_env(self, aou_env):
        d = Dsub(docker_image="img", use_aou_docker_prefix=False)
        assert d.project == "test-project-id"

    def test_user_name_no_dots(self, aou_env):
        d = Dsub(docker_image="img", use_aou_docker_prefix=False)
        assert "." not in d.user_name

    @pytest.mark.aou
    def test_log_path_includes_bucket(self, aou_env):
        d = Dsub(docker_image="img", use_aou_docker_prefix=False)
        assert "gs://test-bucket" in d.log_file_path

    def test_stdout_path_derived_from_log(self, dsub):
        assert dsub.job_stdout.endswith("-stdout.log")

    def test_stderr_path_derived_from_log(self, dsub):
        assert dsub.job_stderr.endswith("-stderr.log")

    def test_custom_log_path(self, aou_env):
        d = Dsub(docker_image="img", log_file_path="gs://bucket/custom.log", use_aou_docker_prefix=False)
        assert d.log_file_path == "gs://bucket/custom.log"

    def test_preemptible_false_by_default(self, dsub):
        assert dsub.preemptible is False

    def test_job_id_empty_before_run(self, dsub):
        assert dsub.job_id == ""


# ---------------------------------------------------------------------------
# Base script generation
# ---------------------------------------------------------------------------

class TestDsubBaseScript:
    def test_contains_dsub_command(self, dsub):
        assert dsub.dsub_base_script().startswith("dsub")

    def test_contains_provider(self, dsub):
        assert "google-batch" in dsub.dsub_base_script()

    def test_contains_machine_type(self, dsub):
        assert "c2d-highcpu-4" in dsub.dsub_base_script()

    def test_contains_image_no_prefix(self, dsub):
        assert "phetk/phetk:latest" in dsub.dsub_base_script()

    def test_aou_prefix_applied_when_enabled(self, aou_env):
        d = Dsub(docker_image="phetk/phetk:latest", use_aou_docker_prefix=True)
        assert "us-docker.pkg.dev/test-project/test-repo" in d.dsub_base_script()

    def test_no_aou_prefix_when_disabled(self, aou_env):
        d = Dsub(docker_image="phetk/phetk:latest", use_aou_docker_prefix=False)
        assert "us-docker.pkg.dev" not in d.dsub_base_script()

    def test_contains_region(self, dsub):
        assert "us-central1" in dsub.dsub_base_script()

    def test_contains_job_name(self, dsub):
        assert dsub.job_name in dsub.dsub_base_script()

    def test_contains_log_path(self, dsub):
        assert dsub.log_file_path in dsub.dsub_base_script()


# ---------------------------------------------------------------------------
# Full dsub script assembly
# ---------------------------------------------------------------------------

class TestDsubScript:
    def test_input_flag_included(self, aou_env):
        d = Dsub(
            docker_image="img",
            input_dict={"INPUT_FILE": "gs://bucket/in.tsv"},
            use_aou_docker_prefix=False,
        )
        assert "--input INPUT_FILE=gs://bucket/in.tsv" in d._dsub_script()

    def test_output_flag_included(self, aou_env):
        d = Dsub(
            docker_image="img",
            output_dict={"OUTPUT_FILE": "gs://bucket/out.tsv"},
            use_aou_docker_prefix=False,
        )
        assert "--output OUTPUT_FILE=gs://bucket/out.tsv" in d._dsub_script()

    def test_env_flag_included(self, aou_env):
        d = Dsub(
            docker_image="img",
            env_dict={"MY_VAR": "my_value"},
            use_aou_docker_prefix=False,
        )
        assert "--env MY_VAR" in d._dsub_script()

    def test_preemptible_flag_when_enabled(self, aou_env):
        d = Dsub(docker_image="img", preemptible=True, use_aou_docker_prefix=False)
        assert "--preemptible" in d._dsub_script()

    def test_no_preemptible_flag_by_default(self, dsub):
        assert "--preemptible" not in dsub._dsub_script()

    def test_disk_type_flag(self, aou_env):
        d = Dsub(docker_image="img", disk_type="pd-ssd", use_aou_docker_prefix=False)
        script = d._dsub_script()
        assert "--disk-type" in script
        assert "pd-ssd" in script

    def test_no_disk_type_flag_by_default(self, dsub):
        assert "--disk-type" not in dsub._dsub_script()

    def test_job_script_flag_included(self, aou_env):
        d = Dsub(docker_image="img", job_script_name="run.sh", use_aou_docker_prefix=False)
        assert "--script run.sh" in d._dsub_script()

    def test_no_script_flag_when_none(self, dsub):
        assert "--script" not in dsub._dsub_script()

    def test_use_private_address_included_by_default(self, dsub):
        assert "--use-private-address" in dsub._dsub_script()

    def test_script_attribute_set_after_call(self, dsub):
        dsub._dsub_script()
        assert dsub.script != ""


# ---------------------------------------------------------------------------
# Custom argument merging
# ---------------------------------------------------------------------------

class TestMergeCustomArgs:
    def test_appends_new_arg(self):
        result = Dsub._merge_custom_args("dsub --regions us-central1", "--timeout 3600")
        assert "--timeout 3600" in result

    def test_overrides_existing_arg(self):
        result = Dsub._merge_custom_args("dsub --regions us-central1", "--regions us-east1")
        assert "us-east1" in result
        assert result.count("--regions") == 1

    def test_empty_custom_args_returns_base(self):
        base = "dsub --regions us-central1"
        assert Dsub._merge_custom_args(base, "") == base

    def test_none_custom_args_returns_base(self):
        base = "dsub --regions us-central1"
        assert Dsub._merge_custom_args(base, None) == base


# ---------------------------------------------------------------------------
# Job status parsing
# ---------------------------------------------------------------------------

class TestCheckJobStatus:
    def test_success_detected(self, dsub):
        stdout = "status: success\nlast-update: '2024-01-01 10:00:00.000'\nstatus-detail: ok"
        _, has_success, has_failed, has_canceled, _, _ = dsub._check_job_status(stdout)
        assert has_success is True
        assert has_failed is False
        assert has_canceled is False

    def test_failed_detected(self, dsub):
        stdout = "status: failed\nlast-update: '2024-01-01 10:00:00.000'\nstatus-detail: error"
        _, _, has_failed, _, _, _ = dsub._check_job_status(stdout)
        assert has_failed is True

    def test_canceled_detected(self, dsub):
        stdout = "status: canceled\nlast-update: '2024-01-01 10:00:00.000'"
        _, _, _, has_canceled, _, _ = dsub._check_job_status(stdout)
        assert has_canceled is True

    def test_empty_stdout_returns_no_terminal_state(self, dsub):
        status, success, failed, canceled, update, detail = dsub._check_job_status("")
        assert status == ""
        assert success is False
        assert failed is False
        assert canceled is False

    def test_last_update_microseconds_stripped(self, dsub):
        stdout = "status: running\nlast-update: '2024-01-01 10:00:00.123456'"
        _, _, _, _, last_update, _ = dsub._check_job_status(stdout)
        assert "." not in last_update

    def test_status_detail_extracted(self, dsub):
        stdout = "status: running\nlast-update: '2024-01-01'\nstatus-detail: Pulling image"
        _, _, _, _, _, detail = dsub._check_job_status(stdout)
        assert detail == "Pulling image"

    def test_completed_is_success(self, dsub):
        stdout = "status: completed\nlast-update: '2024-01-01'"
        _, has_success, _, _, _, _ = dsub._check_job_status(stdout)
        assert has_success is True

    def test_error_is_failed(self, dsub):
        stdout = "status: error\nlast-update: '2024-01-01'"
        _, _, has_failed, _, _, _ = dsub._check_job_status(stdout)
        assert has_failed is True


# ---------------------------------------------------------------------------
# Verily Workbench detection (_utils.is_verily_workbench)
# ---------------------------------------------------------------------------

class TestIsVerilyWorkbench:
    def test_returns_true_when_wb_succeeds(self, mock_subprocess):
        result = MagicMock(returncode=0, stdout="{}", stderr="")
        mock_subprocess["utils"].return_value = result
        assert _utils.is_verily_workbench() is True

    def test_returns_false_when_wb_returns_nonzero(self, mock_subprocess):
        result = MagicMock(returncode=1, stdout="", stderr="error")
        mock_subprocess["utils"].return_value = result
        assert _utils.is_verily_workbench() is False

    def test_returns_false_when_wb_not_installed(self, mock_subprocess):
        mock_subprocess["utils"].side_effect = FileNotFoundError("wb not found")
        assert _utils.is_verily_workbench() is False

    def test_returns_false_on_timeout(self, mock_subprocess):
        mock_subprocess["utils"].side_effect = subprocess.TimeoutExpired("wb", 10)
        assert _utils.is_verily_workbench() is False

    def test_result_is_cached(self, mock_subprocess):
        result = MagicMock(returncode=0, stdout="{}", stderr="")
        mock_subprocess["utils"].return_value = result
        assert _utils.is_verily_workbench() is True
        # Second call should not shell out again.
        mock_subprocess["utils"].reset_mock()
        assert _utils.is_verily_workbench() is True
        mock_subprocess["utils"].assert_not_called()


# ---------------------------------------------------------------------------
# dsub version check (_check_dsub_version)
# ---------------------------------------------------------------------------

class TestCheckDsubVersion:
    def test_no_warning_when_version_matches(self, mock_subprocess, capsys):
        result = MagicMock(
            stdout=f"dsub version {_PINNED_DSUB_VERSION}\n",
            stderr="",
            returncode=0,
        )
        mock_subprocess["dsub"].return_value = result
        _check_dsub_version()
        captured = capsys.readouterr()
        assert "Warning" not in captured.out

    def test_warns_on_wrong_version(self, mock_subprocess, capsys):
        result = MagicMock(
            stdout="dsub version 0.4.13\n",
            stderr="",
            returncode=0,
        )
        mock_subprocess["dsub"].return_value = result
        _check_dsub_version()
        captured = capsys.readouterr()
        assert "Warning" in captured.out
        assert "0.4.13" in captured.out
        assert _PINNED_DSUB_VERSION in captured.out
        assert "RESTART" in captured.out

    def test_warns_when_binary_missing(self, mock_subprocess, capsys):
        mock_subprocess["dsub"].side_effect = FileNotFoundError("dsub not found")
        _check_dsub_version()
        captured = capsys.readouterr()
        assert "Warning" in captured.out
        assert "not found" in captured.out
        assert "RESTART" in captured.out

    def test_warns_on_timeout(self, mock_subprocess, capsys):
        mock_subprocess["dsub"].side_effect = subprocess.TimeoutExpired("dsub", 10)
        _check_dsub_version()
        captured = capsys.readouterr()
        assert "Warning" in captured.out
        assert "timed out" in captured.out

    def test_extracts_version_from_stderr(self, mock_subprocess, capsys):
        # Some dsub versions write to stderr instead of stdout.
        result = MagicMock(
            stdout="",
            stderr=f"dsub version {_PINNED_DSUB_VERSION}\n",
            returncode=0,
        )
        mock_subprocess["dsub"].return_value = result
        _check_dsub_version()
        captured = capsys.readouterr()
        assert "Warning" not in captured.out


# ---------------------------------------------------------------------------
# Verily auto-disable of AoU docker prefix in Dsub.__init__
# ---------------------------------------------------------------------------

class TestDsubInitVerilyAutoDisablePrefix:
    def test_prefix_disabled_on_verily(self, aou_env, mock_subprocess, capsys):
        # wb returns success → Verily detected; dsub version matches → no warning
        mock_subprocess["run"].side_effect = _routing_side_effect(wb_returncode=0)
        d = Dsub(docker_image="phetk/phetk:latest", use_aou_docker_prefix=True)
        assert d.use_aou_docker_prefix is False
        captured = capsys.readouterr()
        assert "Verily Workbench detected" in captured.out

    def test_prefix_preserved_on_aou(self, aou_env, mock_subprocess):
        # wb missing → not on Verily; dsub version matches
        def _route(cmd, *args, **kwargs):
            if isinstance(cmd, list) and cmd[:2] == ["wb", "workspace"]:
                raise FileNotFoundError("wb not found")
            if isinstance(cmd, list) and cmd[:1] == ["dsub"] and "--version" in cmd:
                return _make_subprocess_result(
                    stdout=f"dsub version {_PINNED_DSUB_VERSION}\n",
                )
            return _make_subprocess_result()

        mock_subprocess["run"].side_effect = _route
        d = Dsub(docker_image="phetk/phetk:latest", use_aou_docker_prefix=True)
        assert d.use_aou_docker_prefix is True

    def test_prefix_still_false_if_user_passed_false_on_verily(
        self, aou_env, mock_subprocess
    ):
        mock_subprocess["run"].side_effect = _routing_side_effect(wb_returncode=0)
        d = Dsub(docker_image="phetk/phetk:latest", use_aou_docker_prefix=False)
        assert d.use_aou_docker_prefix is False


# ---------------------------------------------------------------------------
# Base script regression tests — lock in AoU + Verily generated flags so we
# don't inadvertently break the dsub command downstream.
# ---------------------------------------------------------------------------

class TestDsubBaseScriptAouRegression:
    """AoU default path — must match the original pre-Verily format."""

    def test_aou_uses_short_network(self, dsub):
        assert '--network "global/networks/network"' in dsub.dsub_base_script()

    def test_aou_uses_short_subnetwork(self, dsub):
        assert '--subnetwork "regions/us-central1/subnetworks/subnetwork"' in dsub.dsub_base_script()

    def test_aou_uses_gcloud_service_account(self, dsub):
        assert '--service-account "$(gcloud config get-value account)"' in dsub.dsub_base_script()

    def test_aou_image_has_prefix_when_enabled(self, aou_env):
        d = Dsub(docker_image="phetk/phetk:latest", use_aou_docker_prefix=True)
        script = d.dsub_base_script()
        assert "us-docker.pkg.dev/test-project/test-repo/phetk/phetk:latest" in script

    def test_aou_image_no_prefix_when_disabled(self, dsub):
        script = dsub.dsub_base_script()
        assert '--image "phetk/phetk:latest"' in script
        assert "us-docker.pkg.dev" not in script


class TestDsubBaseScriptVerilyRegression:
    """Verily path — swaps network/subnetwork/service-account + strips prefix."""

    @pytest.fixture
    def verily_dsub(self, aou_env, mock_subprocess, monkeypatch):
        mock_subprocess["run"].side_effect = _routing_side_effect(wb_returncode=0)
        monkeypatch.setenv("PET_SA_EMAIL", "pet-123@verily-project.iam.gserviceaccount.com")
        return Dsub(docker_image="phetk/phetk:latest", use_aou_docker_prefix=True)

    def test_verily_uses_fully_qualified_network(self, verily_dsub):
        assert (
            '--network "projects/test-project-id/global/networks/network"'
            in verily_dsub.dsub_base_script()
        )

    def test_verily_uses_fully_qualified_subnetwork(self, verily_dsub):
        assert (
            '--subnetwork "projects/test-project-id/regions/us-central1/subnetworks/subnetwork"'
            in verily_dsub.dsub_base_script()
        )

    def test_verily_uses_pet_sa_email(self, verily_dsub):
        assert (
            '--service-account "pet-123@verily-project.iam.gserviceaccount.com"'
            in verily_dsub.dsub_base_script()
        )

    def test_verily_falls_back_to_google_service_account_email(
        self, aou_env, mock_subprocess, monkeypatch
    ):
        mock_subprocess["run"].side_effect = _routing_side_effect(wb_returncode=0)
        monkeypatch.delenv("PET_SA_EMAIL", raising=False)
        monkeypatch.setenv(
            "GOOGLE_SERVICE_ACCOUNT_EMAIL",
            "pet-456@verily-project.iam.gserviceaccount.com",
        )
        d = Dsub(docker_image="phetk/phetk:latest", use_aou_docker_prefix=True)
        assert (
            '--service-account "pet-456@verily-project.iam.gserviceaccount.com"'
            in d.dsub_base_script()
        )

    def test_verily_image_has_no_prefix(self, verily_dsub):
        script = verily_dsub.dsub_base_script()
        assert '--image "phetk/phetk:latest"' in script
        assert "us-docker.pkg.dev" not in script

    def test_verily_base_script_still_contains_core_flags(self, verily_dsub):
        script = verily_dsub.dsub_base_script()
        assert "--provider \"google-batch\"" in script
        assert "--regions \"us-central1\"" in script
        assert "--machine-type \"c2d-highcpu-4\"" in script
        assert "--boot-disk-size 50" in script
        assert "--disk-size 256" in script
