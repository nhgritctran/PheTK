# PheWAS Module

Run phenome-wide association studies to test associations between a variable of interest and multiple phenotypes.

## Running PheWAS

Performs logistic or Cox regression analysis across all phecodes, testing association with an independent variable while adjusting for covariates.

### Key Parameters
- `phecode_version`: Phecode version to use, "1.2" or "X" (str, required)
- `phecode_count_file_path`: Path to phecode counts file (str, required)
- `cohort_file_path`: Path to cohort file with covariates (str, required)
- `covariate_cols`: List of covariate column names (list[str], required)
- `independent_variable_of_interest`: Name of primary variable column (str, required)
- `sex_at_birth_col`: Name of sex column with 0/1 values (str, required)
- `male_as_one`: True if male=1, False if male=0 (bool, default: True)
- `icd_version`: ICD version "US", "WHO", or "custom" (str, default: "US")
- `phecode_map_file_path`: Path to custom phecode mapping file (str, optional)
- `phecode_to_process`: Specific phecodes to analyze (list[str] or str, optional)
- `min_cases`: Minimum cases required to test phecode (int, default: 50)
- `min_phecode_count`: Minimum count to qualify as case (int, default: 2)
- `use_exclusion`: Whether to use phecode exclusion ranges (bool, default: False)
- `method`: Regression method — "logit", "cox", "firth_logit", or "firth_cox" (str, default: "logit")
- `firth_penalty_weight`: Penalty weight for Firth methods (float, default: 0.5)
- `firth_max_iter`: Maximum iterations for Firth methods (int, optional)
- `firth_use_lrt`: Whether to use likelihood ratio test p-values for Firth methods (bool, default: True)
- `batch_size`: Number of phecodes per processing batch (int, optional, default: 1 for logit/firth_logit, 10 for cox/firth_cox)
- `fall_back_to_serial`: Fall back to serial processing if parallel fails (bool, default: False)
- `output_file_path`: Output file path (str, optional)
- `verbose`: Print progress for each phecode (bool, default: False)
- `suppress_warnings`: Suppress convergence warnings (bool, default: True)

### Notebook Example
```python
from phetk.phewas import PheWAS

# Run PheWAS analysis
phewas = PheWAS(
    phecode_version="X",
    phecode_count_file_path="phecode_counts.tsv",
    cohort_file_path="cohort.tsv",
    covariate_cols=["age", "sex", "pc1", "pc2", "pc3"],
    independent_variable_of_interest="genotype",
    sex_at_birth_col="sex",
    min_cases=50,
    min_phecode_count=2,
    output_file_path="phewas_results.tsv"
)
phewas.run()
```

### CLI Example
```bash
phetk phewas \
  --phecode_version "X" \
  --cohort_file_path "cohort.tsv" \
  --phecode_count_file_path "phecode_counts.tsv" \
  --sex_at_birth_col "sex" \
  --covariate_cols age sex pc1 pc2 pc3 \
  --independent_variable_of_interest "genotype" \
  --min_cases 50 \
  --min_phecode_count 2 \
  --output_file_path "phewas_results.tsv"
```

## Cox Regression Parameters

Additional parameters for Cox proportional hazards regression:
- `cox_start_date_col`: Column with start dates (str, optional); optional for Cox.
        Date to exclude participants with pre-existing phenotype from cases of a particular phecode.
- `cox_control_observed_time_col`: Column with censoring time for controls (str, required)
- `cox_phecode_observed_time_col`: Column with time to event for cases (str, required)
- `cox_stratification_col`: Column for stratification (str, optional)
- `cox_fallback_step_size`: Step size for convergence issues (float, optional, default: 0.1)

### Notebook Example
```python
phewas = PheWAS(
    phecode_version="X",
    phecode_count_file_path="phecode_counts.tsv",
    cohort_file_path="cohort.tsv",
    covariate_cols=["age", "sex", "pc1"],
    independent_variable_of_interest="exposure",
    sex_at_birth_col="sex",
    method="cox",
    cox_start_date_col="start_date",
    cox_control_observed_time_col="follow_up_time",
    cox_phecode_observed_time_col="time_to_event"
)
phewas.run()
```

### CLI Example
```bash
phetk phewas \
  --phecode_version "X" \
  --cohort_file_path "cohort.tsv" \
  --phecode_count_file_path "phecode_counts.tsv" \
  --sex_at_birth_col "sex" \
  --covariate_cols age sex pc1 \
  --independent_variable_of_interest "exposure" \
  --min_cases 50 \
  --min_phecode_count 2 \
  --method "cox" \
  --cox_start_date_col "start_date" \
  --cox_control_observed_time_col "follow_up_time" \
  --cox_phecode_observed_time_col "time_to_event" \
  --output_file_path "cox_results.tsv"
```

## Firth Penalized Regression

Firth penalized regression reduces bias in maximum likelihood estimates, particularly for rare phenotypes with small case counts or near-separation. Two Firth methods are available: `firth_logit` (penalized logistic) and `firth_cox` (penalized Cox). Both use the [firthmodels](https://pypi.org/project/firthmodels/) library.

Compared to standard methods, Firth regression:
- Produces less biased effect estimates for rare phenotypes
- Improves convergence when standard methods fail due to separation
- Uses likelihood ratio test (LRT) p-values by default, which are more reliable than Wald p-values for small samples
- Applies a slight penalty that results in moderately wider confidence intervals

### Firth Parameters
- `firth_penalty_weight`: Controls the strength of the Firth penalty (float, default: 0.5). Lower values reduce the penalization.
- `firth_max_iter`: Maximum number of iterations for the Firth optimizer (int, optional). Uses backend defaults if not specified.
- `firth_use_lrt`: Whether to use likelihood ratio test p-values instead of Wald p-values (bool, default: True). LRT p-values are recommended for Firth regression.

### Notebook Example (Firth Logistic)
```python
from phetk.phewas import PheWAS

phewas = PheWAS(
    phecode_version="X",
    phecode_count_file_path="phecode_counts.tsv",
    cohort_file_path="cohort.tsv",
    covariate_cols=["age", "sex", "pc1", "pc2", "pc3"],
    independent_variable_of_interest="genotype",
    sex_at_birth_col="sex",
    min_cases=50,
    min_phecode_count=2,
    method="firth_logit",
    output_file_path="phewas_results_firth_logit.tsv"
)
phewas.run()
```

### Notebook Example (Firth Cox)
```python
phewas = PheWAS(
    phecode_version="X",
    phecode_count_file_path="phecode_counts.tsv",
    cohort_file_path="cohort.tsv",
    covariate_cols=["age", "sex", "pc1"],
    independent_variable_of_interest="exposure",
    sex_at_birth_col="sex",
    method="firth_cox",
    cox_control_observed_time_col="follow_up_time",
    cox_phecode_observed_time_col="time_to_event",
    output_file_path="phewas_results_firth_cox.tsv"
)
phewas.run()
```

### CLI Example
```bash
phetk phewas \
  --phecode_version "X" \
  --cohort_file_path "cohort.tsv" \
  --phecode_count_file_path "phecode_counts.tsv" \
  --sex_at_birth_col "sex" \
  --covariate_cols age sex pc1 pc2 pc3 \
  --independent_variable_of_interest "genotype" \
  --min_cases 50 \
  --min_phecode_count 2 \
  --method "firth_logit" \
  --firth_penalty_weight 0.5 \
  --firth_use_lrt True \
  --output_file_path "firth_logit_results.tsv"
```

## Running PheWAS with dsub

Execute PheWAS analysis using Google Cloud dsub for distributed computing on cloud infrastructure.

NOTE: For Cox regression, a standard or highmem machine should be used. For logistic regression, any machine would work.
For example, machine_type="c2d_highmem_4" for Cox regression and machine_type="c2d_highcpu_4" for logistic regression.

**See [dsub-considerations.md](dsub-considerations.md) for detailed setup, parameter guidance, and useful utilities.**

### Key Parameters
- `docker_image`: Docker image containing PheWAS dependencies (str, required)
- `job_script_name`: Name of bash script to execute (str, default: "phewas_script.sh")
- `job_name`: Custom name for dsub job (str, optional)
- `input_dict`: Mapping of input variables to cloud storage paths (dict, optional)
- `output_dict`: Mapping of output variables to cloud storage paths (dict, optional)
- `env_dict`: Environment variables to set in job (dict, optional)
- `machine_type`: Google Cloud machine type (str, default: "c2d-highcpu-4")
- `boot_disk_size`: Size of boot disk in GB (int, default: 50)
- `disk_size`: Size of additional disk in GB (int, default: 256)
- `region`: Google Cloud region for execution (str, default: "us-central1")
- `provider`: Cloud provider backend (str, default: "google-batch")
- `preemptible`: Whether to use preemptible instances (bool, default: False)
- `use_private_address`: Whether to use private IP addresses (bool, default: True)

### Notebook Example
```python
from phetk.phewas import PheWAS

# Create PheWAS instance
phewas = PheWAS(
    phecode_version="X",
    phecode_count_file_path="gs://your-bucket/phecode_counts.tsv",
    cohort_file_path="gs://your-bucket/cohort.tsv",
    covariate_cols=["age", "sex", "pc1", "pc2", "pc3"],
    independent_variable_of_interest="genotype",
    sex_at_birth_col="sex",
    min_cases=50,
    min_phecode_count=2,
    method="logit",
    output_file_path="gs://your-bucket/phewas_results.tsv"
)

# Run with dsub
phewas.run_dsub(
    docker_image="phetk/phetk:latest",
    job_name="my-phewas-job",
    machine_type="c2d-standard-4",
    region="us-central1",
    preemptible=True
)
```

## Advanced Options

### Process Specific Phecodes
```python
# Single phecode
phecode_to_process="185"

# Multiple phecodes
phecode_to_process=["185", "250.2", "401.1"]
```

### Phecode Exclusion (1.2 only)
```python
use_exclusion=True  # Apply phecode exclusion ranges
```

### Custom Phecode Mapping
```python
icd_version="custom"
phecode_map_file_path="path/to/custom_mapping.tsv"
```

### Parallel Processing
```python
batch_size=10  # Process 10 phecodes per batch
fall_back_to_serial=True  # Use serial if parallel fails
```

## Get Phecode Data

Retrieve cohort data for specific phecode after running PheWAS:

```python
# Get data for phecode "185"
phecode_data = phewas.get_phecode_data("185")
```

Returns dataframe with original cohort data plus `is_phecode_case` column indicating case/control status.

## Output Format

Results file columns vary by method.

### Logistic methods (`logit`, `firth_logit`)
- `phecode`: Phecode tested
- `cases`: Number of cases
- `controls`: Number of controls
- `p_value`: P-value from regression
- `neg_log_p_value`: -log10(p_value)
- `standard_error`: Standard error of beta
- `beta`: Log-odds coefficient
- `conf_int_1`: Lower bound of 95% CI for beta
- `conf_int_2`: Upper bound of 95% CI for beta
- `odds_ratio`: Exponentiated beta
- `log10_odds_ratio`: log10(odds_ratio)
- `converged`: Whether regression converged
- `phecode_sex_restriction`: Sex restriction if applicable
- `phecode_string`: Phecode description
- `phecode_category`: Phecode category

### Cox methods (`cox`, `firth_cox`)
- `phecode`: Phecode tested
- `cases`: Number of cases
- `controls`: Number of controls
- `p_value`: P-value from regression
- `neg_log_p_value`: -log10(p_value)
- `standard_error`: Standard error of log hazard ratio
- `hazard_ratio`: Hazard ratio
- `hazard_ratio_low`: Lower bound of 95% CI for hazard ratio
- `hazard_ratio_high`: Upper bound of 95% CI for hazard ratio
- `log_hazard_ratio`: Log hazard ratio coefficient
- `concordance_index`: Concordance index (C-statistic)
- `stratified_by`: Stratification variable if used
- `convergence`: Whether regression converged
- `phecode_sex_restriction`: Sex restriction if applicable
- `phecode_string`: Phecode description
- `phecode_category`: Phecode category

## Important Notes

- Sex column must contain 0/1 values only
- Include sex in `covariate_cols` if using as covariate
- Non-converged results are kept and flagged in `converged` column
- Minimum case/control requirements prevent spurious associations