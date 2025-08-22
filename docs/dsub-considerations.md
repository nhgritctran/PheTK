# dsub Considerations

This document provides important considerations when running PheWAS analyses using Google Cloud dsub for distributed computing.

## Important
It is recommended that user would get familiar running phetk locally before running with dsub. Troubleshooting dsub is more challenging and time-consuming than running locally.

## General Overview/Considerations

dsub is Google's tool for submitting batch jobs to cloud computing environments. When running PheWAS with dsub, your analysis executes on Google Cloud infrastructure rather than locally, enabling:

- **Scalability**: Handle large cohorts and extensive phecode analyses
- **Cost efficiency**: Pay only for compute time used, with preemptible instance options
- **Resource flexibility**: Choose appropriate machine types for your analysis needs
- **Parallel processing**: Distribute workload across multiple cloud instances

### Prerequisites
- Google Cloud project with appropriate permissions
- Input/output files accessible via Google Cloud Storage (`gs://` URLs)
- Docker image containing PheWAS dependencies

### Google Batch
From July 2025, Google started using Google Batch and dsub has been updated to be compatible with the new provider.
However, there are current issues with running dsub jobs (as of Aug 2025):
- dsub is not able to set boot disk type which causes error with newer machine generations such as c3/c4... Therefore, currently, the best machine generation available is c2.
- dsub job status would still remain for 45-60 days after the dsub worker is terminated. In the past, the job status would be deleted right after worker termination.

### Runtime
If dsub runtime is abnormally longer than normal (job status is `RUNNING`), e.g., 1.5 to 2 times longer than running locally on a comparable machine, user should terminate the job to avoid unexpected cost and investigate the log files.

## dsub Parameter Notes

### Critical Parameters

**`docker_image`** (required)
- Use `"phetk/phetk:latest"` for official PheWAS image
- Ensure the image is accessible from your Google Cloud project
- Custom images must include all PheWAS dependencies

**`machine_type`** 
For any GCP machine generation, there are 3 main types `highcpu`, `standard` (2x `highcpu` RAM), and `highmem` (2x `standard` RAM)
- **Logistic regression**: any machine type should work, e.g., `"c2d-highcpu-4"`
- **Cox regression**: use `standard` or `highmem` machine, e.g., `"c2d-standard-4"` or `"c2d-highmem-4"`

**Storage Configuration**
- `boot_disk_size`: 50GB default, increase for large Docker images
- `disk_size`: 256GB default, increase for large datasets or intermediate files
- `disk_type`: Make sure the disk type is compatible with the machine type used; Refer to Google Cloud documentation for details.

**Cost Optimization**
- `preemptible=True`: Up to 80% cost savings (jobs may be interrupted)
- Choose regions with lower pricing when data locality allows. For example, if your data is in an us-central location, dsub jobs should also be run in an us-central location.

### File Path Requirements
All file paths must use Google Cloud Storage URLs:
```python
# Correct
phecode_count_file_path="gs://your-bucket/phecode_counts.tsv"
cohort_file_path="gs://your-bucket/cohort.tsv"
output_file_path="gs://your-bucket/results/phewas_results.tsv"

# Incorrect - local paths won't work
phecode_count_file_path="./local_file.tsv"
```

## Useful Utilities

Once `run_dsub()` is successfully executed, there are useful utilities that can be used to manage the job.

In the examples below, phewas was instantiated as `phewas`:
```python
from phetk.phewas import PheWAS
phewas = PheWAS(...)
phewas.run_dsub(...)
```

### Job Monitoring
```python
# Check job status (one-time)
phewas.check_status()

# Monitor job with streaming updates
phewas.check_status(mode="stream", update_interval=30)
```

### Log Management
```python
# View recent stdout logs
phewas.view_log(log_type="stdout", n_lines=50)

# View error logs
phewas.view_log(log_type="stderr", n_lines=20)

# View all logs
phewas.view_log(log_type="all")
```

### Job Control
```python
# Cancel specific job
phewas.kill()

# Cancel all your running jobs
phewas.kill_all()

# View all running jobs
phewas.view_all()

# Test dsub configuration
phewas.echo_hello_test()
```