"""
Functional tests for PheWAS with Firth penalized regression methods.
Uses demo-generated synthetic data — no mocking, no AoU required.
"""
import os
import numpy as np
import pytest
import polars as pl

from phetk._utils import generate_mock_phewas_data
from phetk.phewas import PheWAS


LOGIT_COLS = {"phecode", "p_value", "neg_log_p_value", "beta", "cases", "controls",
              "standard_error", "conf_int_1", "conf_int_2", "odds_ratio",
              "log10_odds_ratio", "converged"}
COX_COLS = {"phecode", "p_value", "neg_log_p_value", "hazard_ratio", "cases", "controls",
            "standard_error", "hazard_ratio_low", "hazard_ratio_high",
            "log_hazard_ratio", "concordance_index", "stratified_by", "convergence"}


@pytest.fixture(scope="module")
def demo_files(tmp_path_factory):
    """Generate demo data once for all tests in this module."""
    tmp = tmp_path_factory.mktemp("firth_demo")
    orig = os.getcwd()
    os.chdir(str(tmp))
    try:
        generate_mock_phewas_data(cohort_size=300, phecode="GE_979.2")
    finally:
        os.chdir(orig)
    return {
        "cohort": str(tmp / "example_cohort.tsv"),
        "phecode_counts": str(tmp / "example_phecode_counts.tsv"),
        "tmp": tmp,
    }


@pytest.fixture(scope="module")
def cox_files(demo_files, tmp_path_factory):
    """Augment demo data with columns required for Cox regression."""
    tmp = tmp_path_factory.mktemp("firth_cox")
    np.random.seed(0)

    cohort = pl.read_csv(demo_files["cohort"], separator="\t")
    cohort_path = str(tmp / "cox_cohort.tsv")
    cohort.write_csv(cohort_path, separator="\t")

    phecode_counts = pl.read_csv(
        demo_files["phecode_counts"],
        separator="\t",
        schema_overrides={"phecode": str},
    )
    phecode_path = str(tmp / "cox_phecode.tsv")
    phecode_counts.write_csv(phecode_path, separator="\t")

    return {"cohort": cohort_path, "phecode_counts": phecode_path, "tmp": tmp}


# ---------------------------------------------------------------------------
# Firth logistic regression
# ---------------------------------------------------------------------------

class TestFirthLogit:
    def test_creates_output_file(self, demo_files):
        out = str(demo_files["tmp"] / "firth_logit_results.tsv")
        PheWAS(
            phecode_version="X",
            phecode_count_file_path=demo_files["phecode_counts"],
            cohort_file_path=demo_files["cohort"],
            covariate_cols=["age", "sex", "pc1", "pc2", "pc3"],
            independent_variable_of_interest="independent_variable_of_interest",
            sex_at_birth_col="sex",
            min_cases=10,
            min_phecode_count=2,
            output_file_path=out,
            method="firth_logit",
        ).run()
        assert os.path.exists(out)

    def test_output_has_expected_columns(self, demo_files):
        out = str(demo_files["tmp"] / "firth_logit_cols.tsv")
        PheWAS(
            phecode_version="X",
            phecode_count_file_path=demo_files["phecode_counts"],
            cohort_file_path=demo_files["cohort"],
            covariate_cols=["age", "sex", "pc1", "pc2", "pc3"],
            independent_variable_of_interest="independent_variable_of_interest",
            sex_at_birth_col="sex",
            min_cases=10,
            min_phecode_count=2,
            output_file_path=out,
            method="firth_logit",
        ).run()
        result = pl.read_csv(out, separator="\t", schema_overrides={"phecode": str})
        for col in LOGIT_COLS:
            assert col in result.columns, f"Missing column: {col}"

    def test_firth_logit_columns_match_logit(self, demo_files):
        out_logit = str(demo_files["tmp"] / "std_logit_for_cmp.tsv")
        out_firth = str(demo_files["tmp"] / "firth_logit_for_cmp.tsv")

        PheWAS(
            phecode_version="X",
            phecode_count_file_path=demo_files["phecode_counts"],
            cohort_file_path=demo_files["cohort"],
            covariate_cols=["age", "sex", "pc1", "pc2", "pc3"],
            independent_variable_of_interest="independent_variable_of_interest",
            sex_at_birth_col="sex",
            min_cases=10,
            min_phecode_count=2,
            output_file_path=out_logit,
            method="logit",
        ).run()

        PheWAS(
            phecode_version="X",
            phecode_count_file_path=demo_files["phecode_counts"],
            cohort_file_path=demo_files["cohort"],
            covariate_cols=["age", "sex", "pc1", "pc2", "pc3"],
            independent_variable_of_interest="independent_variable_of_interest",
            sex_at_birth_col="sex",
            min_cases=10,
            min_phecode_count=2,
            output_file_path=out_firth,
            method="firth_logit",
        ).run()

        logit_result = pl.read_csv(out_logit, separator="\t", schema_overrides={"phecode": str})
        firth_result = pl.read_csv(out_firth, separator="\t", schema_overrides={"phecode": str})
        assert set(logit_result.columns) == set(firth_result.columns)

    def test_p_values_between_0_and_1(self, demo_files):
        out = str(demo_files["tmp"] / "firth_logit_pvals.tsv")
        PheWAS(
            phecode_version="X",
            phecode_count_file_path=demo_files["phecode_counts"],
            cohort_file_path=demo_files["cohort"],
            covariate_cols=["age", "sex", "pc1", "pc2", "pc3"],
            independent_variable_of_interest="independent_variable_of_interest",
            sex_at_birth_col="sex",
            min_cases=10,
            min_phecode_count=2,
            output_file_path=out,
            method="firth_logit",
        ).run()
        result = pl.read_csv(out, separator="\t", schema_overrides={"phecode": str})
        p_vals = result["p_value"].drop_nulls().to_list()
        assert all(0 <= p <= 1 for p in p_vals)


# ---------------------------------------------------------------------------
# Firth Cox regression
# ---------------------------------------------------------------------------

class TestFirthCox:
    def test_creates_output_file(self, cox_files):
        out = str(cox_files["tmp"] / "firth_cox_results.tsv")
        PheWAS(
            phecode_version="X",
            phecode_count_file_path=cox_files["phecode_counts"],
            cohort_file_path=cox_files["cohort"],
            covariate_cols=["age", "sex", "pc1", "pc2", "pc3"],
            independent_variable_of_interest="independent_variable_of_interest",
            sex_at_birth_col="sex",
            cox_control_observed_time_col="observed_time",
            cox_phecode_observed_time_col="phecode_observed_time",
            min_cases=10,
            min_phecode_count=2,
            output_file_path=out,
            method="firth_cox",
        ).run()
        assert os.path.exists(out)

    def test_output_has_expected_columns(self, cox_files):
        out = str(cox_files["tmp"] / "firth_cox_cols.tsv")
        PheWAS(
            phecode_version="X",
            phecode_count_file_path=cox_files["phecode_counts"],
            cohort_file_path=cox_files["cohort"],
            covariate_cols=["age", "sex", "pc1", "pc2", "pc3"],
            independent_variable_of_interest="independent_variable_of_interest",
            sex_at_birth_col="sex",
            cox_control_observed_time_col="observed_time",
            cox_phecode_observed_time_col="phecode_observed_time",
            min_cases=10,
            min_phecode_count=2,
            output_file_path=out,
            method="firth_cox",
        ).run()
        result = pl.read_csv(out, separator="\t", schema_overrides={"phecode": str})
        for col in COX_COLS:
            assert col in result.columns, f"Missing column: {col}"

    def test_firth_cox_columns_match_cox(self, cox_files):
        out_cox = str(cox_files["tmp"] / "std_cox_for_cmp.tsv")
        out_firth = str(cox_files["tmp"] / "firth_cox_for_cmp.tsv")

        PheWAS(
            phecode_version="X",
            phecode_count_file_path=cox_files["phecode_counts"],
            cohort_file_path=cox_files["cohort"],
            covariate_cols=["age", "sex", "pc1", "pc2", "pc3"],
            independent_variable_of_interest="independent_variable_of_interest",
            sex_at_birth_col="sex",
            cox_control_observed_time_col="observed_time",
            cox_phecode_observed_time_col="phecode_observed_time",
            min_cases=10,
            min_phecode_count=2,
            output_file_path=out_cox,
            method="cox",
        ).run()

        PheWAS(
            phecode_version="X",
            phecode_count_file_path=cox_files["phecode_counts"],
            cohort_file_path=cox_files["cohort"],
            covariate_cols=["age", "sex", "pc1", "pc2", "pc3"],
            independent_variable_of_interest="independent_variable_of_interest",
            sex_at_birth_col="sex",
            cox_control_observed_time_col="observed_time",
            cox_phecode_observed_time_col="phecode_observed_time",
            min_cases=10,
            min_phecode_count=2,
            output_file_path=out_firth,
            method="firth_cox",
        ).run()

        cox_result = pl.read_csv(out_cox, separator="\t", schema_overrides={"phecode": str})
        firth_result = pl.read_csv(out_firth, separator="\t", schema_overrides={"phecode": str})
        assert set(cox_result.columns) == set(firth_result.columns)
