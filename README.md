# PheTK - The Phenotype Toolkit
The official repository of PheTK, a fast python library for Phenome Wide Association Studies (PheWAS) utilizing both phecode 1.2 and phecodeX 1.0.

__Reference__: Tam C Tran, David J Schlueter, Chenjie Zeng, Huan Mo, Robert J Carroll, Joshua C Denny, PheWAS analysis on large-scale biobank data with PheTK, Bioinformatics, Volume 41, Issue 1, January 2025, btae719, https://doi.org/10.1093/bioinformatics/btae719

__Contact__: [PheTK@mail.nih.gov](mailto:PheTK@mail.nih.gov)

***

## QUICK LINKS
- [Installation](#1-installation)
- [System requirements](#2-system-requirements)
- [1-minute PheWAS demo](#3-1-minute-phewas-demo)
- [PheTK description](#4-descriptions)
- [Usage examples](#5-usage)
  - [Cohort module](docs/cohort-module.md)
  - [Phecode module](docs/phecode-module.md)
  - [PheWAS module](docs/phewas-module.md)
  - [Plot module](docs/plot-module.md)
- Platform specific tutorial(s):
  - ___All of Us___: [APOE PheWAS jupyter notebook](https://workbench.researchallofus.org/workspaces/aou-rw-42a1ea44/chargedemo/analysis/preview/2_PheWAS_with_PheTK_demo.ipynb).
This notebook demonstrates PheTK usage on the _All of Us_ Researcher Workbench with APOE PheWAS as a study example.
Please note that this link is only accessible to _All of Us_ registered users.
- [Changelogs and releases](https://github.com/nhgritctran/PheTK/releases): 
from v0.1.45, please use [GitHub Releases](https://github.com/nhgritctran/PheTK/releases) for the latest versions and changelogs. 
Legacy changelogs were archived in [CHANGELOG.md](legacy/CHANGELOG.md).
- Resource to learn about PheWAS and phecode: [The PheWAS Catalog](https://phewascatalog.org/).

***

## 1. INSTALLATION
The latest version of PheTK can be installed using the pip install command in the terminal (Python 3.7 or newer):

```
pip install phetk --upgrade
```

or Jupyter Notebook (restart kernel after installation and prior to importing):
```
!pip install phetk --upgrade
```

To check current installed version:
```
pip show phetk | grep Version
```

## 2. SYSTEM REQUIREMENTS
PheTK was developed for efficient processing of large data while being resource-friendly. 
It was tested on different platforms from laptops to different cloud environments.

Here are some minimum VM configuration suggestions for _All of Us_ users:
- Cohort module: default General Analysis (4 CPUs, 15GB RAM) or Hail Genomic Analysis 
(main VM and 2 workers of 4 CPUs, 15GB RAM each) VMs should work. 
Hail Genomic Analysis is only needed for the .by_genotype() method.
- Phecode module: a minimum of 8 CPUs, 52GB RAM standard VM should work for current v7 data.
This might require more RAM depending on the user's custom workflow or data - 
usually Jupyter Python kernel would die at if there is not enough memory.
- PheWAS module: default General Analysis VM (4 CPUs, 15GB RAM) should work. 
However, more CPUs would speed up analysis and using low configurations does not necessarily save computing cost
since total runtime would be longer.
- Plot module: default General Analysis VM (4 CPUs, 15GB RAM) should work.

In practice, users could try different available machine configurations to achieve optimal performance and cost-effectiveness.

## 3. 1-MINUTE PHEWAS DEMO

User can run the quick 1-minute PheWAS demo with the following command in a terminal:

```
python3 -m phetk.demo
```

Or in Jupyter Notebook:

```
from phetk import demo

demo.run()
```

The example files (`example_cohort.tsv`, `example_phecode_counts.tsv`, and `example_phewas_results.tsv`) 
generated in this Demo should be in users' current working directory. 
New-to-PheWAS users could explore these files to get a sense of what data are used or generated in PheWAS with PheTK.

## 4. DESCRIPTIONS
PheTK is a fast python library for Phenome Wide Association Studies (PheWAS) utilizing both phecode 1.2 and phecodeX 1.0.

### 4.1. PheWAS workflow and PheTK modules
![PheWAS workflow and PheTK modules](img/readme/PheTK_flowchart.png)
Standard PheWAS workflow. Green texts are PheTK module names. 
Black components are supported while gray ones are not supported by PheTK currently.

### 4.2. PheTK module descriptions
This table will be updated as we update PheTK. 

All modules can be used together or independently, 
e.g., users who only need to run PheWAS analysis can provide their own cohort and phecode count data as input for PheWAS module. 

| Module  | Class   | Method(s)     | Platform    | Requirements/Notes                                                           |
|---------|---------|---------------|-------------|------------------------------------------------------------------------------|
| cohort  | Cohort  | by_genotype   | _All of Us_ | None                                                                         |
|         |         |               | Other       | Variant data stored in Hail matrix table                                     |
|         |         | add_covariate | _All of Us_ | None                                                                         |
|         |         |               | Other       | Google BigQuery OMOP database                                                |
|         |         |               |             | Required tables: person, condition_occurrence, observation, death, & concept |
| phecode | Phecode | count_phecode | _All of Us_ | None                                                                         | 
|         |         |               | Other       | User provided cohort ICD code data                                           |
|         |         |               |             | User can use custom ICD-to-phecode mapping table.                            |
| pheWAS  | PheWAS  | all methods   | Any         | None                                                                         |
| plot    | Plot    | all methods   | Any         | None                                                                         |
| demo    |         | all methods   | Any         | None                                                                         |

_All of Us_: the _All of Us_ Research Program (https://allofus.nih.gov/)

## 5. USAGE

For detailed usage examples and documentation for each module, please refer to the individual module documentation:

- **[Cohort module](docs/cohort-module.md)** - Generate genetic cohorts and add covariates
- **[Phecode module](docs/phecode-module.md)** - Map ICD codes to phecodes and generate counts
- **[PheWAS module](docs/phewas-module.md)** - Run PheWAS analysis with logistic or Cox regression
- **[Plot module](docs/plot-module.md)** - Generate Manhattan plots and other visualizations
