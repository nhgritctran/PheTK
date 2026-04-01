# Cohort Module

Generate genetic cohorts and add covariates for phenome-wide association studies.

## by_genotype

Generate cohort based on genotype of variant of interest. Extracts genotype data from VCF files (default) or Hail matrix tables and creates a cohort file with person IDs and genotype labels. Handles multi-allelic sites automatically.

### Key Parameters
- `chromosome_number`: Chromosome number (int)
- `genomic_position`: Genomic position on chromosome (int)
- `ref_allele`: Reference allele (str)
- `alt_allele`: Alternative allele (str)
- `gt_dict`: Genotype mapping, e.g., `{0: "0/0", 1: ["0/1", "1/1"]}` (dict)
- `reference_genome`: "GRCh37" or "GRCh38" (str, default: "GRCh38")
- `data_format`: Genotype data source — `"vcf"` (default) or `"hail"` (str)
- `call_set`: AoU callset name — `"acaf_threshold"` (default) or `"exome"` (str)
- `data_path`: Override path to genotype data (str, optional). For VCF: path to a `.vcf.gz`/`.vcf.bgz`/`.bcf` file or AoU shard directory. For Hail: path to `.mt` directory. Auto-resolved on AoU if not provided.
- `output_file_path`: Output TSV file path (str, optional)

### Data Formats

#### VCF (default, `data_format="vcf"`)
Uses [pysam](https://pysam.readthedocs.io/) to read tabix-indexed VCF files (included with phetk).

- Supports local and GCS (`gs://`) paths with requester-pays
- On AoU, automatically locates the correct VCF shard via binary search on tabix indices
- Uses the `WGS_ACAF_THRESHOLD_VCF_PATH` environment variable if set, otherwise constructs the path from the CDR version
- For multi-allelic sites, non-target alternate alleles are treated as reference (matching Hail `split_multi_hts` behaviour)

#### Hail (`data_format="hail"`)
Uses Hail to read matrix tables. Requires `pip install phetk[hail]`.

- Requires a Spark environment (e.g., AoU Workbench)
- Splits multi-allelic sites using `hl.split_multi_hts()`
- On AoU, auto-resolves the matrix table path from the `WGS_ACAF_THRESHOLD_MULTI_HAIL_PATH` environment variable

### Notebook Example
```python
from phetk.cohort import Cohort

cohort = Cohort(platform="aou", aou_db_version=8)

# Using VCF (default)
cohort.by_genotype(
    chromosome_number=7,
    genomic_position=117559590,
    ref_allele="ATCT",
    alt_allele="A",
    gt_dict={0: "0/0", 1: ["0/1", "1/1"]},
    output_file_path="cftr_cohort.tsv"
)

# Using Hail
cohort.by_genotype(
    chromosome_number=7,
    genomic_position=117559590,
    ref_allele="ATCT",
    alt_allele="A",
    gt_dict={0: "0/0", 1: ["0/1", "1/1"]},
    data_format="hail",
    output_file_path="cftr_cohort_hail.tsv"
)

# Using a custom VCF file path
cohort.by_genotype(
    chromosome_number=7,
    genomic_position=117559590,
    ref_allele="ATCT",
    alt_allele="A",
    gt_dict={0: "0/0", 1: ["0/1", "1/1"]},
    data_format="vcf",
    data_path="gs://my-bucket/my_variants.vcf.bgz",
    output_file_path="cftr_cohort_custom.tsv"
)
```

### CLI Example
```bash
# Using VCF (default)
phetk cohort by-genotype \
  --chromosome_number 7 \
  --genomic_position 117559590 \
  --ref_allele "ATCT" \
  --alt_allele "A" \
  --gt_dict '{"0": "0/0", "1": ["0/1", "1/1"]}' \
  --output_file_path "cftr_cohort.tsv"

# Using Hail
phetk cohort by-genotype \
  --chromosome_number 7 \
  --genomic_position 117559590 \
  --ref_allele "ATCT" \
  --alt_allele "A" \
  --gt_dict '{"0": "0/0", "1": ["0/1", "1/1"]}' \
  --data_format hail \
  --output_file_path "cftr_cohort_hail.tsv"
```

**Note:** In the CLI, `--gt_dict` accepts a JSON string (keys must be strings, converted to integers internally). In Python, pass a regular dict with integer keys.

CLI JSON examples:
- Single genotype per group: `'{"0": "0/0", "1": "1/1"}'`
- Multiple genotypes per group: `'{"0": "0/0", "1": ["0/1", "1/1"]}'`
- Three groups: `'{"0": "0/0", "1": "0/1", "2": "1/1"}'`

## add_covariates

Add demographic, clinical, and genetic covariates to existing cohort. Merges covariate data from database with input cohort file.

### Key Parameters
- `cohort_file_path`: Path to cohort CSV/TSV file with person_id column (str, required)
- `date_of_birth`: Include date of birth (bool, default: False)
- `year_of_birth`: Include year of birth (bool, default: False)
- `current_age`: Include current age (bool, default: False)
- `current_age_squared`: Include current age squared (bool, default: False)
- `current_age_cubed`: Include current age cubed (bool, default: False)
- `sex_at_birth`: Include sex at birth (bool, default: True)
- `last_ehr_date`: Include date of last diagnosis event (bool, default: False)
- `age_at_last_ehr_event`: Include age at last diagnosis event (bool, default: False)
- `age_at_last_ehr_event_squared`: Include age at last diagnosis event squared (bool, default: False)
- `age_at_last_ehr_event_cubed`: Include age at last diagnosis event cubed (bool, default: False)
- `ehr_length`: Include EHR record length in years (bool, default: False)
- `dx_code_occurrence_count`: Include diagnosis code occurrence count (bool, default: False)
- `dx_condition_count`: Include unique diagnosis condition count (bool, default: False)
- `genetic_ancestry`: Include predicted ancestry (bool, default: False)
- `first_n_pcs`: Number of genetic PCs to include (int, default: 0)
- `chunk_size`: Participant IDs per processing thread (int, default: 10000)
- `drop_nulls`: Drop rows with null values (bool, default: False)
- `output_file_path`: Output TSV file path (str, optional)

### Notebook Example
```python
from phetk.cohort import Cohort

# Create cohort instance
cohort = Cohort(platform="aou", aou_db_version=8)

# Add covariates to cohort
cohort.add_covariates(
    cohort_file_path="cftr_cohort.tsv",
    sex_at_birth=True,
    age_at_last_ehr_event=True,
    first_n_pcs=10,
    drop_nulls=True,
    output_file_path="cohort_with_covariates.tsv"
)
```

### CLI Example
```bash
phetk cohort add-covariates \
  --cohort_file_path "cftr_cohort.tsv" \
  --sex_at_birth true \
  --age_at_last_ehr_event true \
  --first_n_pcs 10 \
  --drop_nulls true \
  --output_file_path "cohort_with_covariates.tsv"
```

## Platform Configuration

### All of Us
```python
cohort = Cohort(platform="aou", aou_db_version=8)
```
- `platform`: "aou" for All of Us
- `aou_db_version`: CDR version (6-8, default: 8)
- `aou_omop_cdr`: Optional CDR string for OMOP data

### Custom Platform
```python
cohort = Cohort(platform="custom", gbq_dataset_id="your_dataset_id")
```
- `platform`: "custom" for non-All of Us platforms
- `gbq_dataset_id`: Google BigQuery dataset ID (required for custom)

## Covariate Descriptions

- `date_of_birth`: Participant date of birth
- `year_of_birth`: Birth year only
- `current_age`: Current age or age at death
- `current_age_squared/cubed`: Polynomial age terms for modeling
- `sex_at_birth`: Sex at birth from survey data (0/1)
- `last_ehr_date`: Date of last diagnosis event in EHR
- `age_at_last_ehr_event`: Age at last diagnosis event in EHR
- `age_at_last_ehr_event_squared/cubed`: Polynomial age terms
- `ehr_length`: Number of years EHR record spans
- `dx_code_occurrence_count`: Count of diagnosis code occurrences on unique dates
- `dx_condition_count`: Count of unique diagnosis conditions
- `genetic_ancestry`: Predicted ancestry (e.g., "eur", "afr")
- `first_n_pcs`: First n genetic principal components from PCA