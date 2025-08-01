# Tutorial Guide - Read This First!

## Getting Started

**Important:** Please start with the **Logistic Regression notebook** as it contains comprehensive explanations of PheTK's core concepts and workflow. The other notebooks are more concise and focus on demonstrating specific features in different contexts.

Note that these notebooks must be run on the _All of Us_ Researcher Workbench since certain functions are customized for it.
That said, they should be good practical examples of how to use PheTK.

## Available Tutorials

ALl the notebooks were run in a 16CPU 104GB RAM VM on the _All of Us_ Researcher Workbench.

### 1. [APOE PheWAS - Logistic Regression](v0.2.1%20APOE%20PheWAS%20-%20Logistic%20Regression%20-%20release%20candidate.ipynb)
This comprehensive tutorial walks through a complete PheWAS analysis using logistic regression, including detailed explanations of data preparation, covariate selection, and result visualizations.

### 2. [APOE PheWAS - Cox Regression](v0.2.1%20APOE%20PheWAS%20-%20Cox%20Regression%20-%20release%20candidate.ipynb)
Demonstrates how to perform survival analysis using Cox proportional hazards regression for time-to-event phenotype data.

### 3. [APOE PheWAS - CLI](v0.2.1%20APOE%20PheWAS%20-%20CLI%20-%20release%20candidate.ipynb)
Shows how to use PheTK's command-line interface for running PheWAS analyses directly from the terminal.

### 4. [APOE PheWAS - dsub](v0.2.1%20APOE%20PheWAS%20-%20dsub%20-%20release%20candidate.ipynb)
Illustrates how to run distributed PheWAS analyses on Google Cloud Platform using PheTK's dsub wrapper for simple, scalable, and efficient processing.

## Notes

The tutorial notebooks demonstrate how to use PheTK in specific contexts and study designs. For comprehensive documentation on the package and detailed information on each module's functions, please refer to:

- [Main README](../../README.md) - Package overview and installation
- [Cohort Module](../cohort-module.md) - Genotype-based cohort generation and covariate addition
- [Phecode Module](../phecode-module.md) - Phecode mapping utilities
- [PheWAS Module](../phewas-module.md) - Phenome-wide association studies
- [Plot Module](../plot-module.md) - Visualization tools for PheWAS results