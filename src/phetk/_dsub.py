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
        :type docker_image: str
        :param job_script_name: Path to the script file to execute in the job
        :type job_script_name: str
        :param job_name: Name for the job (auto-generated if None)
        :type job_name: str | None
        :param input_dict: Dictionary mapping input variable names to GCS paths
        :type input_dict: dict | None
        :param output_dict: Dictionary mapping output variable names to GCS paths
        :type output_dict: dict | None
        :param env_dict: Dictionary of environment variables to set in the job
        :type env_dict: dict | None
        :param log_file_path: Custom path for log files (auto-generated if None)
        :type log_file_path: str | None
        :param machine_type: GCP machine type to use for the job
        :type machine_type: str
        :param disk_type: Type of disk to use (None for default)
        :type disk_type: str | None
        :param boot_disk_size: Size of boot disk in GB
        :type boot_disk_size: int
        :param disk_size: Size of additional disk in GB
        :type disk_size: int
        :param user_project: Google Cloud project for billing
        :type user_project: str
        :param project: Google Cloud project to run the job in
        :type project: str
        :param dsub_user_name: Username for dsub job identification
        :type dsub_user_name: str
        :param user_name: Username for job naming and identification
        :type user_name: str
        :param bucket: Google Cloud Storage bucket for logs and data
        :type bucket: str
        :param google_project: Google Cloud project ID
        :type google_project: str
        :param region: GCP region to run the job in
        :type region: str
        :param provider: Dsub provider to use (google-batch, google-v2, etc.)
        :type provider: str
        :param preemptible: Whether to use preemptible instances
        :type preemptible: bool
        :param use_private_address: Whether to use private IP addresses
        :type use_private_address: bool
        :param custom_args: Additional custom arguments for dsub command
        :type custom_args: str | None
        :param use_aou_docker_prefix: Whether to prepend AoU artifact registry prefix
        :type use_aou_docker_prefix: bool
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
        self.dsub_start_time = None
        self.dsub_end_time = None
        self.dsub_runtime = None

    def _dsub_script(self) -> str:
        """
        Generate the dsub command script with all configured parameters.
        
        :return: Complete dsub command as a string
        :rtype: str
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

    def check_status(
        self, 
        full: bool = False, 
        custom_args: str | None = None, 
        streaming: bool = False, 
        update_interval: int = 10,
        verbose: bool = False
    ) -> None:
        """
        Check the status of the submitted job using dstat command.
        
        :param full: Whether to show full detailed status information
        :type full: bool
        :param custom_args: Additional custom arguments for dstat command
        :type custom_args: str | None
        :param streaming: Whether to continuously monitor status with auto-refresh
        :type streaming: bool
        :param update_interval: Seconds between status updates when streaming
        :type update_interval: int
        :param verbose: Whether to print debug information for status detection
        :type verbose: bool
        :return: None
        :rtype: None
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
            except ImportError:
                pass

            last_status = ""
            last_check_time = 0
            
            # Print initial runtime line
            print(f"Refresh interval: {update_interval}s | Runtime: Initializing...", end="", flush=True)
            
            while True:
                current_time = time.time()
                
                # Calculate runtime based on dsub job start time
                if self.dsub_start_time is not None:
                    runtime = datetime.datetime.now() - self.dsub_start_time
                    runtime_str = str(runtime).split('.')[0]  # Remove microseconds
                else:
                    runtime_str = "Unknown (job not started)"
                
                # Check status only at specified intervals
                if current_time - last_check_time >= update_interval:
                    last_check_time = current_time
                    
                    # Run command and capture output
                    result = subprocess.run([check_status], shell=True, capture_output=True, text=True)
                    current_status = result.stdout.strip()

                    # Check for terminal states using full status
                    status_value = ""
                    if result.stdout:
                        # Get full status to check actual job status line
                        full_status_cmd = check_status + " --full"
                        full_result = subprocess.run([full_status_cmd], shell=True, capture_output=True, text=True)

                        # Look for status line in full output
                        if full_result.stdout:
                            for line in full_result.stdout.split('\n'):
                                if line.strip().lower().startswith('status:'):
                                    status_value = line.split(':', 1)[1].strip().lower()
                                    break

                        if status_value:
                            # Check for success patterns
                            success_patterns = ["success", "succeeded", "complete", "completed", "finished", "done"]
                            failed_patterns = ["unsuccessful", "incomplete", "failed", "error", "failure", "timeout"]
                            canceled_patterns = ["aborted", "terminated", "cancelled", "canceled", "delete", "deleted"]

                            has_success = any(pattern in status_value for pattern in success_patterns)
                            has_failed = any(pattern in status_value for pattern in failed_patterns)
                            has_canceled = any(pattern in status_value for pattern in canceled_patterns)

                            if verbose:
                                print(f"\nDEBUG - Status value: '{status_value}'")
                                print(
                                    f"DEBUG - has_success: {has_success}, has_failed: {has_failed}, has_canceled: {has_canceled}")
                                print()

                            if has_success:
                                self.dsub_end_time = datetime.datetime.now()
                                if self.dsub_start_time is not None:
                                    self.dsub_runtime = self.dsub_end_time - self.dsub_start_time
                                print()
                                print("\nJob completed successfully!")
                                print()
                                break

                            # Check for failure patterns
                            if has_failed:
                                self.dsub_end_time = datetime.datetime.now()
                                if self.dsub_start_time is not None:
                                    self.dsub_runtime = self.dsub_end_time - self.dsub_start_time
                                print()
                                print("\nJob failed!")
                                print()
                                break

                            # Check for canceled/deleted patterns
                            if has_canceled:
                                self.dsub_end_time = datetime.datetime.now()
                                if self.dsub_start_time is not None:
                                    self.dsub_runtime = self.dsub_end_time - self.dsub_start_time
                                print()
                                print("\nJob was canceled or deleted!")
                                print()
                                break
                    
                    # Check for empty status (worker shutdown)
                    if not current_status and self.dsub_start_time is not None and (datetime.datetime.now() - self.dsub_start_time).total_seconds() > 60:
                        print("\r" + " " * 80)  # Clear current line
                        print("\rNo job status found - worker has likely shut down")
                        break
                    
                    # Check if status changed
                    if current_status != last_status:
                        # Clear current runtime line and replace with new status
                        print("\r" + " " * 80)  # Clear current line
                        current_time_str = datetime.datetime.now().strftime("%H:%M:%S")
                        print(f"\r[{current_time_str}] Job Status: {status_value.upper()}\n{current_status}")
                        print()
                        last_status = current_status
                        # Print new runtime line below the status
                        print(f"Refresh interval: {update_interval}s | Runtime: {runtime_str}", end="", flush=True)
                    else:
                        # Update runtime line in place
                        print(f"\rRefresh interval: {update_interval}s | Runtime: {runtime_str}", end="", flush=True)
                else:
                    # Just update runtime display every second
                    print(f"\rRefresh interval: {update_interval}s | Runtime: {runtime_str}", end="", flush=True)

                # Sleep for 1 second to create smooth runtime updates
                time.sleep(1)
        else:
            subprocess.run([check_status], shell=True)

    def view_log(self, log_type: str = "stdout", n_lines: int = 10) -> None:
        """
        View the job logs from Google Cloud Storage.
        
        :param log_type: Type of log to view ('stdout', 'stderr', or 'full')
        :type log_type: str
        :param n_lines: Number of lines to display from the log file
        :type n_lines: int
        :return: None
        :rtype: None
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

    def kill(self) -> None:
        """
        Kill/cancel the running job using ddel command.
        
        Note: Requires that the job has been submitted and job_id is available.
        
        :return: None
        :rtype: None
        """
        kill_job = (
            f"ddel  --users \"{self.user_name}\" --project {self.project} --jobs \"{self.job_id}\""
        )
        subprocess.run([kill_job], shell=True)

    def kill_all(self) -> None:
        """
        Kill/cancel all running jobs using ddel command.

        :return: None
        :rtype: None
        """
        kill_jobs = (
            f"ddel --users \"{self.user_name}\" --project {self.project} --jobs \"*\" "
        )
        subprocess.run([kill_jobs], shell=True)

    def run(self, show_command: bool = False, timeout: int = 60) -> None:
        """
        Submit and run the dsub job on Google Cloud Platform.
        
        :param show_command: Whether to display the dsub command being executed
        :type show_command: bool
        :param timeout: Maximum time in seconds to wait for job submission
        :type timeout: int
        :return: None
        :rtype: None
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
                self.dsub_start_time = datetime.datetime.now()  # Record job start time
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
