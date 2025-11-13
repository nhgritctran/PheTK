# PheTK - The Phenotype Toolkit
The official repository of PheTK, a fast python library for Phenome Wide Association Studies (PheWAS) utilizing both phecode 1.2 and phecodeX 1.0.

__Reference__: Tam C Tran, David J Schlueter, Chenjie Zeng, Huan Mo, Robert J Carroll, Joshua C Denny, PheWAS analysis on large-scale biobank data with PheTK, Bioinformatics, Volume 41, Issue 1, January 2025, btae719, https://doi.org/10.1093/bioinformatics/btae719

__Contact__: [PheTK@mail.nih.gov](mailto:PheTK@mail.nih.gov)

## 🆕 WHAT'S NEW IN v0.2
Major updates in this release:
- **Cox regression support** - Added survival analysis capabilities alongside logistic regression
- **dsub integration** - Built-in support for distributed computing on Google Cloud Platform
- **Forest plot visualization** - New main visualization option alongside Manhattan plots
- **PEP-compliant naming** - Changed to lowercase package/module names (affects import syntax)
- **Expanded CLI support** - Added command-line interfaces for cohort and phecode modules
- **Simplified CLI commands** - Added entry points for easier CLI usage (e.g., `phetk phewas` instead of `python3 -m phetk.phewas`)
- **Enhanced user experience** - Various improvements for clarity and usability

[**📋 View full changelog**](https://github.com/nhgritctran/PheTK/releases)

Version 0.1.47 is the last stable version of version 0.1. 
Users can still continue to use this version, and the previous README file can be found [here](legacy/README_legacy.md)  

***

## QUICK LINKS
- [Installation](#1-installation)
- [1-minute PheWAS demo](#2-1-minute-phewas-demo)
- [PheTK description](#3-descriptions)
- [Usage examples](#4-usage)
  - [Cohort module](docs/cohort-module.md)
  - [Phecode module](docs/phecode-module.md)
  - [PheWAS module](docs/phewas-module.md)
  - [Plot module](docs/plot-module.md)
- [System requirements & computing resources](#5-system-requirements--computing-resources)
- Platform specific tutorial(s):
  - ___All of Us___: [Tutorial notebooks](docs/tutorials/README_FIRST.md) - Interactive Jupyter notebooks demonstrating PheTK usage on the _All of Us_ Researcher Workbench with various analysis examples.
Please note that all examples require _All of Us_ registered user access.
- [Changelogs and releases](https://github.com/nhgritctran/PheTK/releases): 
from v0.1.45, please use [GitHub Releases](https://github.com/nhgritctran/PheTK/releases) for the latest versions and changelogs. 
Legacy changelogs were archived in [CHANGELOG.md](legacy/CHANGELOG.md).
- Resource to learn about PheWAS and phecode: [The PheWAS Catalog](https://phewascatalog.org/).

***

## 1. INSTALLATION

### Using pip

The latest version (v0.2+) of PheTK can be installed using the pip install command in the terminal
(note that the lowercase package name "phetk" starts from version 0.2+):

```
pip install phetk --upgrade
```

Users can also specify a version, e.g., for the last stable version of version 0.1:

```
pip install PheTK==0.1.47
```

To check current installed version:
```
pip show phetk | grep Version
```

### Using Docker
```bash
docker pull phetk/phetk:latest
```

## 2. 1-MINUTE PHEWAS DEMO

User can run the quick 1-minute PheWAS demo with the following command in a terminal:

```
phetk demo
```

Or in Jupyter Notebook:

```
from phetk import demo

demo.run()
```

The example files (`example_cohort.tsv`, `example_phecode_counts.tsv`, and `example_phewas_results.tsv`) 
generated in this Demo should be in users' current working directory. 
New-to-PheWAS users could explore these files to get a sense of what data are used or generated in PheWAS with PheTK.

## 3. DESCRIPTIONS
PheTK is a fast python library for Phenome Wide Association Studies (PheWAS) utilizing both phecode 1.2 and phecodeX 1.0.

![PheWAS workflow and PheTK modules](img/readme/PheTK_flowchart.png)
Standard PheWAS workflow. Green italicized texts are PheTK module names. 
Black components are supported while gray ones are not supported by PheTK currently.

_All of Us_: the _All of Us_ Research Program (https://allofus.nih.gov/)

## 4. USAGE

For detailed usage examples and documentation for each module, please refer to the individual module documentation:

- **[Cohort module](docs/cohort-module.md)** - Generate genetic cohorts and add covariates
- **[Phecode module](docs/phecode-module.md)** - Map ICD codes to phecodes and generate counts
- **[PheWAS module](docs/phewas-module.md)** - Run PheWAS analysis with logistic or Cox regression
- **[Plot module](docs/plot-module.md)** - Generate Manhattan plots and other visualizations

## 5. SYSTEM REQUIREMENTS

PheTK was developed for efficient processing of large data while being resource-friendly. 
It was tested on different platforms from laptops to different cloud environments.

### General Requirements
PheTK's resource requirements vary by usage context. The information in this section is tailored towards cloud computing platforms where large biobanks are often hosted.
- All PheTK functions run on standard machines, except `by_genotype()` in the Cohort module which requires a Spark cluster (dataproc VM)
- Both logistic regression and Cox regression scale with CPU counts for faster processing. See figure S2 below from PheTK publication for more information.
In our experience, 4 CPU machines are the most cost-efficient, especially for large-scale analyses.
- For an end-to-end pipeline, the system requirements should be based on the most demanding steps. 
For example, for the _All of Us_ data v8, a VM with 16CPU 104GB RAM and 2 dataproc workers at default settings should work;
if users only need to run PheWAS analysis, it can be run at a much lower configuration as shown in figure S2.

![PheTK Performance Benchmarks](img/readme/FigureS2.png)
**Figure S2**: Logistic regression performance benchmarks from PheTK publication showing scalability with different CPU configurations and cohort sizes.

#### PheWAS Module - Logistic Regression
- **Minimal resources required** - Can run efficiently on lightweight configurations
- **Minimum tested configuration**: GCP `X-highcpu-4` (4 vCPUs, 8GB RAM, X=GCP machine type, e.g., c2d) or equivalent
- Uses multithreading for parallel processing with lower memory overhead

#### PheWAS Module - Cox Regression  
- **Slightly higher resources required** - Uses multiprocessing which demands more memory
- **Minimum tested configuration**: GCP `X-standard-4` (4 vCPUs, 16GB RAM, X=GCP machine type, e.g., c2d) or equivalent
- The additional memory accommodates the multiprocessing overhead for survival analysis

#### Phecode Module (ICD Code Mapping)
- **Memory requirements scale with cohort size** - Large cohorts require higher memory configurations
- **Recommended**: For _All of Us_ database v8 with over 500k participants, phecode mapping could be done with a 16 vCPU 104GB RAM machine.

