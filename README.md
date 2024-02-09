# PheTK - The Phenotype Toolkit
The official repository of PheTK.

## 1. INSTALLATION
PheTK can be installed using pip install command in a terminal (Python 3.7 or newer):

```
pip install PheTK
```

or Jupyter Notebook (restart kernel after installation and prior to importing):
```
!pip install PheTK
```

## 2. SYSTEM REQUIREMENTS
As PheTK utilizes multithreading to speed up computational processes, the more CPUs/cores the system has, 
the faster it would run.
Memory requirement would vary based on user actual workflow and how efficient it is.

In our test, a PheWAS for a cohort of 400,000+ participants and 12 covariates successfully completed on a 
MacBook Pro M2 Pro 12 CPU cores 32GB RAM in ; 
PheWAS for the same cohort took ~12 minutes with a cloud instance of 96 CPUs 370GB RAM 
(memory choice here is mainly for redundancy and for comparison with other tests that we did).

CPU/RAM configurations and pipeline runtime also vary by computing platforms and operating systems.

In short, the more CPUs the better and RAM should be adequate for user workflow.

## 3. 1-MINUTE PHEWAS DEMO

User can run the quick 1-minute PheWAS demo with the following command in a terminal:

```
python3 -m PheTK.Demo
```

Or in Jupyter Notebook:

```
from PheTK import Demo

Demo.run()
```

## 4. DESCRIPTIONS
PheTK is a fast and efficient python library for Phenome Wide Association Studies (PheWAS) and other analyses 
utilizing both phecode 1.2 and phecodeX 1.0.

### 4.1. PheWAS workflow and PheTK modules
![PheWAS workflow and PheTK modules](img/readme/PheTK_flowchart.png)
Standard PheWAS workflow. Green texts are PheTK module names. Black and gray components are steps supported or 
not supported by PheTK currently.

### 4.2. PheTK module descriptions
This table will be updated as we update PheTK.

| Module  | Class   | Method(s)     | Platform  | Requirements/Notes                                                           |
|---------|---------|---------------|-----------|------------------------------------------------------------------------------|
| Cohort  | Cohort  | by_genotype   | All of Us | None                                                                         |
|         |         |               | Other     | Variant data stored in Hail matrix table                                     |
|         |         | add_covariate | All of Us | None                                                                         |
|         |         |               | Other     | Google BigQuery OMOP database                                                |
|         |         |               |           | Required tables: person, condition_occurrence, observation, death, & concept |
| Phecode | Phecode | count_phecode | All of Us | None                                                                         | 
|         |         |               | Other     | User provided cohort ICD code data                                           |
|         |         |               |           | User can use custom ICD-to-phecode mapping table.                            |
| PheWAS  | PheWAS  | all methods   | Any       | None                                                                         |
| Plot    | Plot    | all methods   | Any       | None                                                                         |
| Demo    |         | all methods   | Any       | None                                                                         |

All of Us: the All of Us Research Program (https://allofus.nih.gov/)

## 5. USAGE
As shown in module descriptions, some features of Cohort and Phecode modules are optimized to support the data 
structure of the All of Us Research Program. 
PheWAS, Plot, and Demo modules can be run on any platform.

### 5.1. Cohort module
Cohort module can be used for generating genetic cohort and add certain covariates to a cohort.

#### 5.1.1. by_genotype

This function takes genetic variant information as input, 
and generates cohort with matching genotypes as an output csv file.

For example, we generate cohort for _CFTR_ variant chr7-117559590-ATCT-A with 
heterozygous (0/1 genotype) participants as cases and homozygous reference (0/0 genotype) participants as controls.

#### Jupyter Notebook example for All of US Researcher Workbench:
For All of Us data version 7, the default Hail matrix table is the ACAF (common variant) table.
User can use a different table by providing table location in the mt_path parameter.
```
from PheTK.Cohort import Cohort

# instantiate class Cohort object for All of Us database version 7
cohort = Cohort(platform="aou", aou_db_version=7)

# generate cohort by genotype
cohort.by_genotype(
    chromosome_number=7,
    genomic_position=117559590,
    ref_allele="ATCT",
    alt_allele="A",
    case_gt="0/1",
    control_gt="0/0",
    reference_genome="GRCh38",
    mt_path=None,
    output_file_name="cftr_cohort.csv"
)
```

#### Jupyter Notebook example for other platforms:
For other platforms, user need to provide the location of Hail matrix table file for mt_path parameter.
```
from PheTK.Cohort import Cohort

# instantiate class Cohort object for All of Us database version 7
cohort = Cohort(platform="custom")

# generate cohort by genotype
cohort.by_genotype(
    chromosome_number=7,
    genomic_position=117559590,
    ref_allele="ATCT",
    alt_allele="A",
    case_gt="0/1",
    control_gt="0/0",
    reference_genome="GRCh38",
    mt_path="/path/to/hail_matrix_table.mt",
    output_file_name="cftr_cohort.csv"
)
```

#### 5.1.2. add_covariates
This function is currently customized for the All of Us Research Platform. 
t takes a cohort csv file and covariate selection as input, 
and generate a new cohort csv file with covariate data added as output. 
Input cohort data which must have at least "person_id" column.

In this example, we are adding age at last diagnosis event, sex at birth and 10 genetic PCs (provided by All of Us).
These options were set to True (or 10 in case of first_n_pcs).

The covariates shown in this example are currently supported by PheTK. Users should only change parameter value to True 
for covariates to be used in subsequent PheWAS. All parameters are set to False by default, i.e., user only need to 
specify parameters of interest.

#### Jupyter Notebook example for All of US Researcher Workbench:
```
# user can skip the import and instantiation steps if running continuously 
# from previous by_genotype example, i.e., skip directly to add covariates step.
from PheTK.Cohort import Cohort

# instantiate class Cohort object for All of Us database version 7
cohort = Cohort(platform="aou", aou_db_version=7)

# RUN EITHER LONG OR SHORT VERSION BELOW
# add covariates - long version, including all currently supported covariate options
cohort.add_covariates(
    cohort_csv_path="aou_chr7_117559590_ATCT_A.csv",
    natural_age=False,
    age_at_last_event=True,
    sex_at_birth=True,
    ehr_length=False,
    dx_code_occurrence_count=False,
    dx_condition_count=False,
    genetic_ancestry=False,
    first_n_pcs=10,
    drop_nulls=True,
    output_file_name="cohort_with_covariates.csv"
)

# add covariates - short version, i.e., users do not need to list unused covariates
cohort.add_covariates(
    cohort_csv_path="aou_chr7_117559590_ATCT_A.csv",
    age_at_last_event=True,
    sex_at_birth=True,
    first_n_pcs=10,
    drop_nulls=True,
    output_file_name="cohort_with_covariates.csv"
)
```

### 5.2. Phecode module
Phecode module is used to retrieve ICD code data of participants, map ICD codes to phecode 1.2 or phecodeX 1.0, 
and aggregate the counts for each phecode of each participant.

The ICD code retrieval is done automatically for All of Us platform when users instantiate class Phecode.
For other platforms, users must provide your own ICD code data.

Example of ICD code data: 
- Each row must be unique, i.e., there should not be 2 instances of 1 ICD codes in the same day.
- Data must have these exact column names. Date column is not used for later ICD-to-phecode mapping, 
but should be included to make sure ICD events occurred on unique dates.

| person_id | date      | vocabulary_id | ICD   |
|-----------|-----------|---------------|-------|
| 13579     | 1-11-2010 | ICD9CM        | 786.2 |
| 13579     | 1-31-2010 | ICD9CM        | 786.2 |
| 13579     | 12-4-2017 | ICD10CM       | R05.1 |
| 24680     | 3-12-2012 | ICD9CM        | 659.2 |
| 24680     | 4-18-2018 | ICD10CM       | R50   |

#### Jupyter Notebook example for All of Us:
```
from PheTK.Phecode import Phecode

phecode = Phecode(platform="aou")
phecode.count_phecode(
    phecode_version="X", 
    icd_version="US",
    phecode_map_file_path=None, 
    output_file_name="my_phecode_counts.csv"
)
```

#### Jupyter Notebook example for other platforms
```
from PheTK.Phecode import Phecode

phecode = Phecode(platform="custom", icd_file_path="/path/to/my_icd_data.csv")
phecode.count_phecode(
    phecode_version="X", 
    icd_version="US", 
    phecode_map_file_path=None,
    output_file_name="my_phecode_counts.csv"
)
```

In both examples, users can provide their own phecode mapping file by adding a csv file path to phecode_map_file_path.

### 5.3. PheWAS module
For new users, it is recommended to run Demo example above and have a look at example cohort and phecode counts file to 
be familiar with input data format. The example files should be generated in user's current working directory.

PheWAS module can be used in both Linux command line interface (CLI) and any Python environment, e.g., 
Jupyter Notebook/Lab.

#### CLI example:
```
python3 -m PheTK.PheWAS \
--phecode_version X \
--cohort_csv_path example_cohort.csv \
--phecode_count_csv_path example_phecode_counts.csv \
--sex_at_birth_col sex \
--covariates age sex pc1 pc2 pc3 
--independent_variable_of_interest independent_variable_of_interest \
--min_case 50 \
--min_phecode_count 2 \
--output_file_name example_phewas_results.csv
```

#### Jupyter Notebook example:
```
from PheTK.PheWAS import PheWAS

example_phewas = PheWAS(
    phecode_version="X",
    phecode_count_csv_path="example_phecode_counts.csv",
    cohort_csv_path="example_cohort.csv",
    sex_at_birth_col="sex",
    covariate_cols=["age", "sex", "pc1", "pc2", "pc3"],
    independent_variable_of_interest="independent_variable_of_interest",
    min_cases=50,
    min_phecode_count=2,
    output_file_name="example_phewas_results.csv"
)
example_phewas.run()
```

## 6. CONTACT

PheTK@mail.nih.gov

## 7. CITATION

TBD