## Changelog:

*___NOTE___: These are legacy changelogs for version v0.1.45 or earlier.
Please use [GitHub Releases](https://github.com/nhgritctran/PheTK/releases) for later versions.*

___version 0.1.45 (01 Oct 2024)___:
- Fixed a minor issue where Bonferroni line is shifted in Manhattan Plot after non-converged phecodes are removed.
- Updated README ([this section](#54-plot-module)) to add description for other plotting features, 
such as setting custom Bonferroni value or custom color palette.

***

___version 0.1.44 (30 Sep 2024)___:
- Removed polars version requirement of <=0.20.26 since polars has fixed multithreading bug.
- Updated Demo with more descriptive texts.
- Updated Plot module (see [this section](#54-plot-module) for usage examples):
  - Added parameter `converged_only` that can be used when instantiate `Plot` class to plot only converged phecodes.
  - Added parameter `marker_size_by_beta` and `marker_scale_factor` in `manhattan()` function
- Updated README to reflect changes.

***

___version 0.1.43 (08 Aug 2024) - IMPORTANT BUG FIX___: 
- Fixed an issue in `.by_genotype()` in Cohort module which might generate incorrect cohort by genotype for
multi-allelic sites from _All of Us_ variant data or custom Hail matrix table. 
For example, for a site that has 3 alleles \["A", "G", "C"] (reference allele, alt allele 1, alt allele 2, as displayed in Hail), 
if user specifies "A" as `ref_allele`, "C" as `alt_allele`, and "0/1" as `case_gt`:
  - Before this fix, given above user inputs, participants with A-C genotype would not be assigned as cases, 
  since the allele index of "C" is still "2", and therefore A-C genotype would be encoded as "0/2", even after multi-allelic split.
  - After this fix, given above user inputs, participants with A-C genotype will be correctly assigned as cases, 
  since the allele index for "C" would be properly updated to "1" after  multi-allelic split.

- This issue affects users who used method `.by_genotype()` to generate cohort:
  - from _All of Us_ data, having ALL the criteria below:
    - the genomic position was a multi-allelic site,
    - the alternative allele of interest was NOT the first alternative allele ("G" in the above example).
  - from custom unsplit matrix table or improperly split matrix table as input, having the same above criteria.

- Going forward, there is nothing changed in how user would use `.by_genotype()`,
i.e., "0" represents reference allele, and "1" represents alternative allele of interest.

- Users should uninstall any previous version, reinstall PheTK, and make sure current version is v0.1.43.
It is recommended for ___affected___ users to rerun `.by_genotype()` step and potentially subsequent steps.

***

___version 0.1.42 (17 Jul 2024):___
- Added method `.get_phecode_data()` in PheWAS class. This method would generate cohort data from input data for a phecode of interest.
Please refer to [this section](#get_phecode_data) in PheWAS module for usage example. 

***

___version 0.1.41 (22 May 2024):___
- Added polars version <= 0.20.26 as dependency requirement since polars version 0.20.27 could cause multithreading issue.

***

___version 0.1.40 (14 May 2024):___
- Fixed an incorrect printout text check in `.by_genotype()` in Cohort module when a variant is not found. 
This is only an aesthetic fix to avoid confusion, and does not affect previously generated cohorts.

***

___version 0.1.39 (02 May 2024):___
- Added lxml as a required dependency during installation as it might not be preinstalled in some platforms.

***

___version 0.1.38 (11 Apr 2024):___
- Updated default _All of Us_ common variant matrix table (ACAF) file path used by `.by_genotype()` method in Cohort module
with _All of Us_ updated path.
- Updated README to add some minimum VM configuration suggestions.

***

___version 0.1.37 (04 Apr 2024):___
- Removed SNOMED codes from SQL query (_ehr_dx_code_query_) generating _ehr_length_, _dx_code_occurrence_count_, 
_dx_condition_count_, and _age_at_last_event_ covariates in `.add_covariates()` method in Cohort module.
  - This was to make it consistent with ICD event query for phecode mapping, 
    i.e., only ICD9CM and ICD10CM would be used as vocabulary_id for these queries.
  - For _All of Us_ users, this change should affect less than 2% of covariate data previously generated 
  by `.add_covariates()` method from version 0.1.36 or earlier, and should not significantly change previous analysis results.
