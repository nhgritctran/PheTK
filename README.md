# PheTK - Phenotype Toolkit
The official repository of PheTK.

## Installation
PheTK can be installed using pip install command in a terminal (Python 3.7 or newer):

```
pip install PheTK
```

## Quick PheWAS demo

User can run the quick 1-minute PheWAS demo with the following command in a terminal:

```
python3 -m PheTK.Demo
```

Or in Jupyter Notebook:

```
from PheTK import Demo

Demo.run()
```

## Description
PheTK is a fast and efficient python library for Phenome Wide Association Studies (PheWAS) and other analyses 
utilizing both phecode 1.2 and phecodeX 1.0.

### PheWAS workflow and PheTK modules
![PheWAS workflow and PheTK modules](img/readme/PheTK_flowchart.png)
Standard PheWAS workflow. Green texts are PheTK module names. Black and gray components are steps supported or 
not supported by PheTK currently.

### PheTK module descriptions
This table will be updated as we update PheTK.

CLI = Command line interface

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

## Usage
As shown in module descriptions, some features of Cohort and Phecode modules are optimized to support the data 
structure of the All of Us Research Program. PheWAS, and Plot modules can be run on any platform.

Below are usage examples of PheTK modules.

### Cohort module

#### by_genotype
Generate cohort for _CFTR_ variant chr7-117559590-ATCT-A with heterozygous (0/1 genotype) participants as cases and 
homozygous reference (0/0 genotype) participants as controls.

##### Jupyter Notebook example for All of US Researcher Workbench:
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

##### Jupyter Notebook example for other platforms:
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
    mt_path="path/to/matrix_table.mt",
    output_file_name="cftr_cohort.csv"
)
```

#### add_covariates


### PheWAS module
PheWAS module can be used in both Linux command line interface (CLI) and Python interactive environment, e.g., Jupyter
Notebook/Lab.

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

## Contact: 

PheTK@mail.nih.gov

## Citation: 

TBD