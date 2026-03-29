"""
Unit tests for _dsub.py — test Dsub construction, command generation, and argument merging.
No actual GCP/dsub calls are made.
"""
import pytest
from phetk._dsub import Dsub


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
