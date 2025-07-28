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
- `method`: "logit" for logistic or "cox" for Cox regression (str, default: "logit")
- `batch_size`: Number of phecodes per processing batch (int, default: 1)
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
- `cox_start_date_col`: Column with study start dates (str, required for Cox)
- `cox_control_observed_time_col`: Column with censoring time for controls (str)
- `cox_phecode_observed_time_col`: Column with time to event for cases (str)
- `cox_stratification_col`: Column for stratification (str)
- `cox_fallback_step_size`: Step size for convergence issues (float, default: 0.1)

### Cox Example
```python
phewas = PheWAS(
    phecode_version="X",
    phecode_count_file_path="phecode_counts.tsv",
    cohort_file_path="cohort.tsv",
    covariate_cols=["age", "sex", "pc1"],
    independent_variable_of_interest="exposure",
    sex_at_birth_col="sex",
    method="cox",
    cox_start_date_col="study_start_date",
    cox_control_observed_time_col="follow_up_time",
    cox_phecode_observed_time_col="time_to_event"
)
phewas.run()
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

Results file contains:
- `phecode`: Phecode tested
- `phecode_string`: Phecode description
- `beta`/`hazard_ratio`: Effect estimate
- `SE`: Standard error
- `p_value`: P-value from regression
- `n_cases`: Number of cases
- `n_controls`: Number of controls
- `converged`: Whether regression converged
- `phecode_sex`: Sex restriction if applicable

## Important Notes

- Sex column must contain 0/1 values only
- Include sex in `covariate_cols` if using as covariate
- Non-converged results are kept and flagged in `converged` column
- Minimum case/control requirements prevent spurious associations