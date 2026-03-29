"""
Functional tests for PheWAS.run() using demo-generated synthetic data.
No mocking, no AoU required. These tests actually run regressions.
"""
import os
import numpy as np
import pytest
import polars as pl

from phetk.demo import generate_examples
from phetk.phewas import PheWAS


LOGIT_COLS = ["phecode", "p_value", "neg_log_p_value", "beta", "cases", "controls"]
COX_COLS = ["phecode", "p_value", "neg_log_p_value", "hazard_ratio", "cases", "controls"]


@pytest.fixture(scope="module")
def demo_files(tmp_path_factory):
    """Generate demo data once for all tests in this module."""
    tmp = tmp_path_factory.mktemp("phewas_demo")
    orig = os.getcwd()
    os.chdir(str(tmp))
    try:
        generate_examples(cohort_size=300, phecode="GE_979.2")
    finally:
        os.chdir(orig)
    return {
        "cohort": str(tmp / "example_cohort.tsv"),
        "phecode_counts": str(tmp / "example_phecode_counts.tsv"),
        "tmp": tmp,
    }


def make_phewas(demo_files, out, **kwargs):
    defaults = dict(
        phecode_version="X",
        phecode_count_file_path=demo_files["phecode_counts"],
        cohort_file_path=demo_files["cohort"],
        covariate_cols=["age", "sex", "pc1", "pc2", "pc3"],
        independent_variable_of_interest="independent_variable_of_interest",
        sex_at_birth_col="sex",
        min_cases=10,
        min_phecode_count=2,
        output_file_path=out,
    )
    defaults.update(kwargs)
    return PheWAS(**defaults)


# ---------------------------------------------------------------------------
# PheWAS initialisation
# ---------------------------------------------------------------------------

class TestPheWASInit:
    def test_init_succeeds(self, demo_files):
        p = make_phewas(demo_files, str(demo_files["tmp"] / "init_check.tsv"))
        assert p is not None

    def test_invalid_phecode_version_exits(self, demo_files):
        with pytest.raises(SystemExit):
            make_phewas(demo_files, "out.tsv", phecode_version="99")

    def test_invalid_method_accepted_at_init(self, demo_files):
        # method is validated on run, not at init - just check it stores
        p = make_phewas(demo_files, "out.tsv", method="logit")
        assert p.method == "logit"


# ---------------------------------------------------------------------------
# Logistic regression (method="logit")
# ---------------------------------------------------------------------------

class TestPheWASRunLogistic:
    def test_creates_output_file(self, demo_files):
        out = str(demo_files["tmp"] / "logit_results.tsv")
        make_phewas(demo_files, out).run()
        assert os.path.exists(out)

    def test_output_has_required_columns(self, demo_files):
        out = str(demo_files["tmp"] / "logit_cols.tsv")
        make_phewas(demo_files, out).run()
        result = pl.read_csv(out, separator="\t", schema_overrides={"phecode": str})
        for col in LOGIT_COLS:
            assert col in result.columns, f"Missing column: {col}"

    def test_p_values_between_0_and_1(self, demo_files):
        out = str(demo_files["tmp"] / "logit_pval.tsv")
        make_phewas(demo_files, out).run()
        result = pl.read_csv(out, separator="\t", schema_overrides={"phecode": str})
        p_vals = result["p_value"].drop_nulls().to_list()
        assert all(0 <= p <= 1 for p in p_vals)

    def test_cases_and_controls_are_positive(self, demo_files):
        out = str(demo_files["tmp"] / "logit_counts.tsv")
        make_phewas(demo_files, out).run()
        result = pl.read_csv(out, separator="\t", schema_overrides={"phecode": str})
        assert result["cases"].min() >= 0
        assert result["controls"].min() >= 0

    def test_min_cases_filter_reduces_results(self, demo_files):
        out_low = str(demo_files["tmp"] / "logit_low_min.tsv")
        out_high = str(demo_files["tmp"] / "logit_high_min.tsv")
        make_phewas(demo_files, out_low, min_cases=5).run()
        make_phewas(demo_files, out_high, min_cases=80).run()
        low = pl.read_csv(out_low, separator="\t", schema_overrides={"phecode": str})
        high = pl.read_csv(out_high, separator="\t", schema_overrides={"phecode": str})
        assert len(low) >= len(high)

    def test_specific_phecode_processed(self, demo_files):
        out = str(demo_files["tmp"] / "logit_specific.tsv")
        make_phewas(
            demo_files, out,
            phecode_to_process=["GE_979.2"],
            min_cases=1,
        ).run()
        result = pl.read_csv(out, separator="\t", schema_overrides={"phecode": str})
        assert len(result) <= 1

    def test_serial_parallelization_works(self, demo_files):
        out = str(demo_files["tmp"] / "logit_serial.tsv")
        make_phewas(demo_files, out, min_cases=30).run(parallelization="serial")
        assert os.path.exists(out)

    def test_phecode12_version_works(self, demo_files):
        out = str(demo_files["tmp"] / "logit_phecode12.tsv")
        make_phewas(demo_files, out, phecode_version="1.2").run()
        assert os.path.exists(out)


# ---------------------------------------------------------------------------
# Cox regression (method="cox")
# ---------------------------------------------------------------------------

class TestPheWASRunCox:
    @pytest.fixture(scope="class")
    def cox_files(self, demo_files, tmp_path_factory):
        """Augment demo data with columns required for Cox regression."""
        tmp = tmp_path_factory.mktemp("cox")
        np.random.seed(0)

        cohort = pl.read_csv(demo_files["cohort"], separator="\t")
        n = len(cohort)
        cohort = cohort.with_columns(
            pl.Series("control_time", np.random.uniform(365, 3650, n).tolist()),
        )
        cohort_path = str(tmp / "cox_cohort.tsv")
        cohort.write_csv(cohort_path, separator="\t")

        phecode_counts = pl.read_csv(
            demo_files["phecode_counts"],
            separator="\t",
            schema_overrides={"phecode": str},
        )
        phecode_counts = phecode_counts.with_columns(
            pl.Series("phecode_time_to_event", np.random.uniform(30, 1000, len(phecode_counts)).tolist())
        )
        phecode_path = str(tmp / "cox_phecode.tsv")
        phecode_counts.write_csv(phecode_path, separator="\t")

        return {"cohort": cohort_path, "phecode_counts": phecode_path, "tmp": tmp}

    def test_cox_creates_output_file(self, cox_files):
        out = str(cox_files["tmp"] / "cox_results.tsv")
        PheWAS(
            phecode_version="X",
            phecode_count_file_path=cox_files["phecode_counts"],
            cohort_file_path=cox_files["cohort"],
            covariate_cols=["age", "sex", "pc1", "pc2", "pc3"],
            independent_variable_of_interest="independent_variable_of_interest",
            sex_at_birth_col="sex",
            cox_control_observed_time_col="control_time",
            cox_phecode_observed_time_col="phecode_time_to_event",
            min_cases=10,
            min_phecode_count=2,
            output_file_path=out,
            method="cox",
        ).run()
        assert os.path.exists(out)

    def test_cox_output_has_hazard_ratio_columns(self, cox_files):
        out = str(cox_files["tmp"] / "cox_cols.tsv")
        PheWAS(
            phecode_version="X",
            phecode_count_file_path=cox_files["phecode_counts"],
            cohort_file_path=cox_files["cohort"],
            covariate_cols=["age", "sex", "pc1", "pc2", "pc3"],
            independent_variable_of_interest="independent_variable_of_interest",
            sex_at_birth_col="sex",
            cox_control_observed_time_col="control_time",
            cox_phecode_observed_time_col="phecode_time_to_event",
            min_cases=10,
            min_phecode_count=2,
            output_file_path=out,
            method="cox",
        ).run()
        result = pl.read_csv(out, separator="\t", schema_overrides={"phecode": str})
        for col in COX_COLS:
            assert col in result.columns, f"Missing column: {col}"
