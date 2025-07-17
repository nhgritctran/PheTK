import datetime
import os
import subprocess
import sys
import time
# noinspection PyUnresolvedReferences,PyProtectedMember
from phetk import _utils

class Dsub:
    """
    This class is a wrapper to run dsub on the All of Us researcher workbench.
    Params input_dict and output_dict values must be paths to Google Cloud Storage bucket(s).
    """

    def __init__(
        self,
        docker_image: str,
        job_script_name: str,
        job_name: str = None,
        input_dict: dict = None,
        output_dict: dict = None,
        env_dict: dict = None,
        log_file_path=None,
        machine_type: str = "c4d-highcpu-4",
        disk_type="hyperdisk-balanced",
        boot_disk_size=50,
        disk_size=256,
        user_project=os.getenv("GOOGLE_PROJECT"),
        project=os.getenv("GOOGLE_PROJECT"),
        dsub_user_name=os.getenv("OWNER_EMAIL").split("@")[0],
        user_name=os.getenv("OWNER_EMAIL").split("@")[0].replace(".", "-"),
        bucket=os.getenv("WORKSPACE_BUCKET"),
        google_project=os.getenv("GOOGLE_PROJECT"),
        region="us-central1",
        provider="google-cls-v2",
        preemptible=False,
        custom_args=None,
    ):
        # Standard attributes
        self.docker_image = docker_image
        self.job_script_name = job_script_name
        self.input_dict = input_dict
        self.output_dict = output_dict
        self.env_dict = env_dict
        self.machine_type = machine_type
        self.disk_type = disk_type
        self.boot_disk_size = boot_disk_size
        self.disk_size = disk_size
        self.user_project = user_project
        self.project = project
        self.dsub_user_name = dsub_user_name
        self.user_name = user_name
        self.bucket = bucket
        self.google_project = google_project
        self.region = region
        self.provider = provider
        self.preemptible = preemptible
        self.custom_args = custom_args

        # job_name
        if job_name is None:
            job_name = "phewas_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.job_name = job_name.replace("_", "-")

        # Internal attributes for optional naming conventions
        self.date = datetime.date.today().strftime("%Y%m%d")
        self.time = datetime.datetime.now().strftime("%H%M%S")

        # log file path
        if log_file_path is not None:
            self.log_file_path = log_file_path
        else:
            self.log_file_path = (
                f"{self.bucket}/dsub/logs/{self.job_name}/{self.user_name}/{self.date}/{self.time}/{self.job_name}.log"
            )

        # some reporting attributes
        self.script = ""
        self.dsub_command = ""
        self.job_id = ""
        self.job_stdout = self.log_file_path.replace(".log", "-stdout.log")
        self.job_stderr = self.log_file_path.replace(".log", "-stderr.log")

    def _dsub_script(self):

        base_script = (
            f"dsub" + " " +
            f"--provider \"{self.provider}\"" + " " +
            f"--regions \"{self.region}\"" + " " +
            f"--machine-type \"{self.machine_type}\"" + " " +
            f"--disk-type \"{self.disk_type}\"" + " " +
            f"--boot-disk-size {self.boot_disk_size}" + " " +
            f"--disk-size {self.disk_size}" + " " +
            f"--user-project \"{self.user_project}\"" + " " +
            f"--project \"{self.project}\"" + " " +
            f"--image \"{self.docker_image}\"" + " " +
            f"--network \"network\"" + " " +
            f"--subnetwork \"subnetwork\"" + " " +
            f"--service-account \"$(gcloud config get-value account)\"" + " " +
            f"--user \"{self.dsub_user_name}\"" + " " +
            f"--logging {self.log_file_path} $@" + " " +
            f"--name \"{self.job_name}\"" + " " +
            f"--env GOOGLE_PROJECT=\"{self.google_project}\"" + " "
        )

        # generate input flags
        input_flags = ""
        if self.input_dict is not None:
            for k, v in self.input_dict.items():
                input_flags += f"--input {k}={v}" + " "

        # generate output flag
        output_flags = ""
        if self.output_dict is not None:
            for k, v in self.output_dict.items():
                output_flags += f"--output {k}={v}" + " "

        # generate env flags
        env_flags = ""
        if self.env_dict is not None:
            for k, v in self.env_dict.items():
                env_flags += f"--env {k}=\"{v}\"" + " "

        # job script flag
        job_script = f"--script {self.job_script_name}" + " "

        # combined script
        script = base_script + env_flags + input_flags + output_flags + job_script

        # add preemptible argument if used
        if self.preemptible:
            script += " --preemptible"

        # add custom arguments
        if self.custom_args is not None:
            script += " " + self.custom_args

        # add attribute for convenience
        self.script = script

        return script

    def check_status(self, full=False, custom_args=None, streaming=False, update_interval=10):

        # base command
        check_status = (
            f"dstat --provider {self.provider} --project {self.project} --location {self.region}"
            f" --jobs \"{self.job_id}\" --users \"{self.user_name}\" --status \"*\""
        )

        # full static status
        if full:
            check_status += " --full"

        # custom arguments
        if custom_args is not None:
            check_status += f" {custom_args}"

        if streaming:
            # Auto-detect notebook
            try:
                # noinspection PyUnresolvedReferences
                from IPython.display import clear_output
                is_notebook = True
            except ImportError:
                is_notebook = False

            while True:
                # Clear output
                if is_notebook:
                    clear_output(wait=True)
                else:
                    os.system("clear" if os.name == "posix" else "cls")

                # Run command and print output
                subprocess.run([check_status], shell=True)

                # Wait
                time.sleep(update_interval)
        else:
            subprocess.run([check_status], shell=True)

    def view_log(self, log_type="stdout", n_lines=10):

        tail = f" | head -n {n_lines}"

        if log_type == "stdout":
            full_command = f"gsutil cat {self.job_stdout}" + tail
        elif log_type == "stderr":
            full_command = f"gsutil cat {self.job_stderr}" + tail
        elif log_type == "full":
            full_command = f"gsutil cat {self.log_file_path}" + tail
        else:
            print("log_type must be 'stdout', 'stderr', or 'full'.")
            sys.exit(1)

        subprocess.run([full_command], shell=True)

    def kill(self):

        kill_job = (
            f"ddel --provider {self.provider} --project {self.project} --location {self.region}"
            f" --jobs \"{self.job_id}\" --users \"{self.user_name}\""
        )
        subprocess.run([kill_job], shell=True)

    def run(self, show_command=False, timeout=60):
        process = None
        try:
            process = subprocess.Popen(
                [self._dsub_script()], 
                shell=True, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            stdout, stderr = process.communicate(timeout=timeout)
            
            if process.returncode == 0:
                print(f"Successfully run dsub to schedule job {self.job_name}.")
                self.job_id = stdout.strip()
                print("job-id:", stdout)
                print()
                self.dsub_command = self._dsub_script().replace("--", "\\ \n--")
                if show_command:
                    print("dsub command:")
                    print(self.dsub_command)
            else:
                print(f"Failed to run dsub to schedule job {self.job_name}.")
                print()
                print("Error information:")
                print(stderr)
                self.dsub_command = self._dsub_script().replace("--", "\\ \n--")
                if show_command:
                    print("dsub command:")
                    print(self.dsub_command)
                    
        except subprocess.TimeoutExpired:
            if process:
                process.kill()
            print(f"dsub command timed out after {timeout} seconds")
            print("This may indicate authentication or connectivity issues with GCP")
            self.dsub_command = self._dsub_script().replace("--", "\\ \n--")
            if show_command:
                print("dsub command that timed out:")
                print(self.dsub_command)