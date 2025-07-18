# Cohort Module

Cohort module can be used for generating genetic cohort and add certain covariates to a cohort.

## by_genotype

This function takes genetic variant information as input 
and generates cohort with matching genotypes as an output tsv file.
As this function uses Hail to extract data from Hail matrix tables, it must be run in a compatible environment,
e.g., a dataproc cluster on _All of Us_ researcher workbench or UK Biobank RAP.

For example, we generate cohort for _CFTR_ variant chr7-117559590-ATCT-A with 
homozygous alternative and heterozygous (1/1 and 0/1 genotypes) participants labeled as 1 
and homozygous reference (0/0 genotype) participants labeled as 0.

If there are two genotype labels, it is recommended to use 0 for the baseline/control genotype(s) and 1 for genotype(s) of interest. 
If there are more than two genotype labels, e.g., 0, 1, and 2, the genotype variable will be treated as ordinal variable in PheWAS.

### Jupyter Notebook example for _All of Us_ Researcher Workbench:
For _All of Us_ data version 7 or later, the default Hail matrix table is the ACAF (common variant) table.
User can use a different table by providing table location in the mt_path parameter.

gt_dict can take either string or list as values.

Please make sure that there are no overlapping genotypes between different genotype labels in gt_dict.
For example, gt_dict = {0: ["0/0", "0/1"], 1: ["0/1", "1/1"]} will raise error for having duplicated genotype "0/1" 
between gt_dict[0] and gt_dict[1].

```python
from PheTK.Cohort import Cohort

# instantiate class Cohort object for _All of Us_ database version 8
cohort = Cohort(platform="aou", aou_db_version=8)

# generate cohort by genotype
cohort.by_genotype(
    chromosome_number=7,
    genomic_position=117559590,
    ref_allele="ATCT",
    alt_allele="A",
    gt_dict={0: "0/0", 1: ["0/1", "1/1"]},
    reference_genome="GRCh38",
    mt_path=None,
    output_file_name="cftr_cohort.tsv"
)
```

### Jupyter Notebook example for other platforms:
For other platforms, users need to provide the location of the Hail matrix table file for mt_path parameter.
```python
from PheTK.Cohort import Cohort

# instantiate class Cohort object for _All of Us_ database version 7
cohort = Cohort(platform="custom")

# generate cohort by genotype
cohort.by_genotype(
    chromosome_number=7,
    genomic_position=117559590,
    ref_allele="ATCT",
    alt_allele="A",
    gt_dict={0: "0/0", 1: ["0/1", "1/1"]}
    reference_genome="GRCh38",
    mt_path="/path/to/hail_matrix_table.mt",
    output_file_name="cftr_cohort.tsv"
)
```

## add_covariates
This function is currently customized for the _All of Us_ Research Platform. 
It takes a cohort csv/tsv file and covariate selection as input, 
and generates a new cohort tsv file with covariate data added as output. 
Input cohort data must have a "person_id" column.

For non-_All of Us_ platforms, a Google BigQuery dataset ID must be provided.

In this example, we are adding age at the last diagnosis event, sex at birth, and 10 genetic PCs (provided by _All of Us_).
These options were set to True (or 10 for first_n_pcs).

The covariates shown in this example are currently supported by PheTK. Users should only change the parameter value to True 
for covariates to be used in subsequent PheWAS. All parameters are set to False by default, i.e., users only need to 
specify parameters of interest as shown in the "short version"

It is highly recommended that users should decide which covariates to use for the study based on their data, 
and it is perfectly fine to add or use their own covariate data if necessary. 

### Jupyter Notebook example for _All of Us_ Researcher Workbench:
```python
# user can skip the import and instantiation steps if running continuously 
# from previous by_genotype example, i.e., skip directly to add covariates step.
from PheTK.Cohort import Cohort

# instantiate class Cohort object for _All of Us_ database version 7
cohort = Cohort(platform="aou", aou_db_version=7)

# RUN EITHER LONG OR SHORT VERSION BELOW
# add covariates - long version, including all currently supported covariate options
cohort.add_covariates(
    cohort_file_path="cftr_cohort.tsv",
    current_age=False,
    age_at_last_event=True,
    sex_at_birth=True,
    ehr_length=False,
    dx_code_occurrence_count=False,
    dx_condition_count=False,
    genetic_ancestry=False,
    first_n_pcs=10,
    drop_nulls=True,
    output_file_name="cohort_with_covariates.tsv"
)

# add covariates - short version, i.e., users do not need to list unused covariates
cohort.add_covariates(
    cohort_file_path="cftr_cohort.tsv",
    age_at_last_event=True,
    sex_at_birth=True,
    first_n_pcs=10,
    drop_nulls=True,
    output_file_name="cohort_with_covariates.tsv"
)
```

### Covariate descriptions:
- _current_age_: current age or age at death
- _age_at_last_event_: age at last diagnosis event (ICD or SNOMED) in EHR.
- _sex_at_birth_: sex at birth
- _ehr_length_: EHR duration, in year, from first to last diagnosis code
- _dx_code_occurrence_count_: counts the occurrences of diagnosis codes throughout EHR of each participant.
For example, person 1 having R50 (fever) code on 5 different dates, R05 (cough) code on 3 different dates, 
and R05.1 (acute cough) code on 2 different dates, will have a dx_code_occurrence_count = 10.
- _dx_condition_count_: counts the number of unique conditions occurred throughout EHR of each participant.
For example, for the same person 1 above, the dx_condition_count = 3 (R05 - cough, R05.1 - acute cough, R50 - fever).
- _genetic_ancestry_: returns string values of predicted ancestries, e.g., "eur", "afr", etc. 
These are only useful if users would like to filter data by genetic ancestries. 
- _first_n_pcs_: retrieves first n genetic PC components from genetic PCA data generated by All of Us.
- _drop_nulls_: remove rows containing null values in any column.

### Jupyter Notebook example for other platforms with OMOP data stored in Google BigQuery:
The only difference in this case is that users need to provide dataset ID for the gbq_dataset_id parameter.
The rest should be the same as the above example.
Please make sure the custom database meets the requirements in section 4.2.
```python
from PheTK.Cohort import Cohort

cohort = Cohort(platform="custom", gbq_dataset_id="Google_BigQuery_dataset_id")
```