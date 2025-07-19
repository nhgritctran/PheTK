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
        log_file_path: str = None,
        machine_type: str = "c2d-highcpu-4",
        disk_type: str | None = None,
        boot_disk_size: int = 50,
        disk_size: int = 256,
        user_project: str = os.getenv("GOOGLE_PROJECT"),
        project: str = os.getenv("GOOGLE_PROJECT"),
        dsub_user_name: str = os.getenv("OWNER_EMAIL", "user").split("@")[0],
        user_name: str = os.getenv("OWNER_EMAIL", "user").split("@")[0].replace(".", "-"),
        bucket: str = os.getenv("WORKSPACE_BUCKET"),
        google_project: str = os.getenv("GOOGLE_PROJECT"),
        region: str = "us-central1",
        provider: str = "google-batch",
        preemptible: bool = False,
        use_private_address: bool = True,
        custom_args: str = None,
        use_aou_docker_prefix: bool = True,
    ):
        """
        Initialize a Dsub instance for running jobs on Google Cloud Platform.
        
        :param docker_image: Name of the Docker image to use for the job
        :param job_script_name: Path to the script file to execute in the job
        :param job_name: Name for the job (auto-generated if None)
        :param input_dict: Dictionary mapping input variable names to GCS paths
        :param output_dict: Dictionary mapping output variable names to GCS paths
        :param env_dict: Dictionary of environment variables to set in the job
        :param log_file_path: Custom path for log files (auto-generated if None)
        :param machine_type: GCP machine type to use for the job
        :param disk_type: Type of disk to use (None for default)
        :param boot_disk_size: Size of boot disk in GB
        :param disk_size: Size of additional disk in GB
        :param user_project: Google Cloud project for billing
        :param project: Google Cloud project to run the job in
        :param dsub_user_name: Username for dsub job identification
        :param user_name: Username for job naming and identification
        :param bucket: Google Cloud Storage bucket for logs and data
        :param google_project: Google Cloud project ID
        :param region: GCP region to run the job in
        :param provider: Dsub provider to use (google-batch, google-v2, etc.)
        :param preemptible: Whether to use preemptible instances
        :param use_private_address: Whether to use private IP addresses
        :param custom_args: Additional custom arguments for dsub command
        :param use_aou_docker_prefix: Whether to prepend AoU artifact registry prefix
        """
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
        self.use_private_address = use_private_address
        self.custom_args = custom_args
        self.use_aou_docker_prefix = use_aou_docker_prefix

        # job_name
        if job_name is None:
            job_name = "phewas-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        # ensure it starts with a letter
        if not job_name[0].isalpha():
            job_name = "job-" + job_name
        self.job_name = job_name.replace("_", "-").lower()

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
        """
        Generate the dsub command script with all configured parameters.
        
        :return: Complete dsub command as a string
        """

        if self.use_aou_docker_prefix:
            aou_docker_prefix = os.getenv("ARTIFACT_REGISTRY_DOCKER_REPO")
            image_tag = f"{aou_docker_prefix}/{self.docker_image}"
        else:
            image_tag = self.docker_image
        base_script = (
            f"dsub" + " " +
            f"--provider \"{self.provider}\"" + " " +
            f"--regions \"{self.region}\"" + " " +
            f"--machine-type \"{self.machine_type}\"" + " " +
            f"--boot-disk-size {self.boot_disk_size}" + " " +
            f"--disk-size {self.disk_size}" + " " +
            f"--user-project \"{self.user_project}\"" + " " +
            f"--project \"{self.project}\"" + " " +
            f"--image \"{image_tag}\"" + " " +
            f"--network \"global/networks/network\"" + " " +
            f"--subnetwork \"regions/{self.region}/subnetworks/subnetwork\"" + " " +
            f"--service-account \"$(gcloud config get-value account)\"" + " " +
            f"--user \"{self.dsub_user_name}\"" + " " +
            f"--logging {self.log_file_path} $@" + " " +
            f"--name \"{self.job_name}\"" + " " +
            f"--env GOOGLE_PROJECT=\"{self.google_project}\"" + " "
        )

        # add disk-type
        disk_type_flag = ""
        if self.disk_type is not None:
            disk_type_flag = f"--disk-type \"{self.disk_type}\"" + " "

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
        script = base_script + disk_type_flag + env_flags + input_flags + output_flags + job_script

        # add preemptible argument if used
        if self.preemptible:
            script += " --preemptible"

        # add use-private-address if used
        if self.use_private_address:
            script += " --use-private-address"

        # add custom arguments
        if self.custom_args is not None:
            script += " " + self.custom_args

        # add attribute for convenience
        self.script = script

        return script

    def check_status(self, full=False, custom_args=None, streaming=False, update_interval=10):
        """
        Check the status of the submitted job using dstat command.
        
        :param full: Whether to show full detailed status information
        :param custom_args: Additional custom arguments for dstat command
        :param streaming: Whether to continuously monitor status with auto-refresh
        :param update_interval: Seconds between status updates when streaming
        """

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
                clear_output = None

            print(f"Refresh interval: {update_interval}s")
            print()

            last_status = ""
            start_time = datetime.datetime.now()
            while True:
                # Calculate runtime
                runtime = datetime.datetime.now() - start_time
                runtime_str = str(runtime).split('.')[0]  # Remove microseconds
                
                # Run command and capture output
                result = subprocess.run([check_status], shell=True, capture_output=True, text=True)
                current_status = result.stdout.strip()
                
                # Update display (either when status changes or to show runtime)
                if current_status != last_status or True:  # Always update to show runtime
                    if is_notebook:
                        # Clear previous output in notebook
                        clear_output(wait=True)
                    current_time = datetime.datetime.now().strftime("%H:%M:%S")
                    print(f"Refresh interval: {update_interval}s | Runtime: {runtime_str}")
                    print(f"[{current_time}] Job Status:")
                    print(current_status)
                    if not is_notebook:
                        print()  # Extra line only for CLI
                    last_status = current_status
                
                # Check for terminal states
                if result.stdout:
                    status_text = result.stdout.lower()
                    if "success" in status_text or "succeeded" in status_text:
                        print("\nJob completed successfully!")
                        break
                    elif "failed" in status_text or "failure" in status_text or "error" in status_text:
                        print("\nJob failed!")
                        break

                # Wait
                time.sleep(update_interval)
        else:
            subprocess.run([check_status], shell=True)

    def view_log(self, log_type="stdout", n_lines=10):
        """
        View the job logs from Google Cloud Storage.
        
        :param log_type: Type of log to view ('stdout', 'stderr', or 'full')
        :param n_lines: Number of lines to display from the log file
        """

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
        """
        Kill/cancel the running job using ddel command.
        
        Note: Requires that the job has been submitted and job_id is available.
        """

        kill_job = (
            f"ddel --provider {self.provider} --project {self.project} --location {self.region}"
            f" --jobs \"{self.job_id}\" --users \"{self.user_name}\""
        )
        subprocess.run([kill_job], shell=True)

    def run(self, show_command=False, timeout=60):
        """
        Submit and run the dsub job on Google Cloud Platform.
        
        :param show_command: Whether to display the dsub command being executed
        :param timeout: Maximum time in seconds to wait for job submission
        """
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