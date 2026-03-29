"""
Integration tests for PheWAS end-to-end on real AoU data.
Skipped automatically outside AoU (WORKSPACE_CDR env var not set).
Run inside AoU with: pytest -m aou

Prerequisites: run cohort + phecode modules first to generate input files,
or set the file paths below to existing files in your workspace.
"""
import os
import pytest
import polars as pl

from phetk.phewas import PheWAS
from phetk.plot import Plot

aou = pytest.mark.skipif(
    not os.getenv("WORKSPACE_CDR"),
    reason="Requires AoU Workbench environment (WORKSPACE_CDR not set)"
)

# ---------------------------------------------------------------------------
# Adjust these paths to point at real cohort / phecode count files
# that were generated inside the AoU workbench.
# ---------------------------------------------------------------------------
COHORT_FILE = os.getenv("TEST_COHORT_FILE", "cohort.tsv")
PHECODE_COUNT_FILE = os.getenv("TEST_PHECODE_COUNT_FILE", "phecode_counts.tsv")
COVARIATE_COLS = ["age", "sex", "pc1", "pc2", "pc3"]
INDEPENDENT_VAR = "independent_variable_of_interest"
SEX_COL = "sex"


@pytest.mark.aou
class TestPheWASAouLogistic:
    def test_run_completes(self, tmp_path):
        out = str(tmp_path / "aou_logit_results.tsv")
        PheWAS(
            phecode_version="X",
            phecode_count_file_path=PHECODE_COUNT_FILE,
            cohort_file_path=COHORT_FILE,
            covariate_cols=COVARIATE_COLS,
            independent_variable_of_interest=INDEPENDENT_VAR,
            sex_at_birth_col=SEX_COL,
            min_cases=50,
            min_phecode_count=2,
            output_file_path=out,
            method="logit",
        ).run()
        assert os.path.exists(out)

    def test_output_has_required_columns(self, tmp_path):
        out = str(tmp_path / "aou_logit_cols.tsv")
        PheWAS(
            phecode_version="X",
            phecode_count_file_path=PHECODE_COUNT_FILE,
            cohort_file_path=COHORT_FILE,
            covariate_cols=COVARIATE_COLS,
            independent_variable_of_interest=INDEPENDENT_VAR,
            sex_at_birth_col=SEX_COL,
            min_cases=50,
            min_phecode_count=2,
            output_file_path=out,
        ).run()
        result = pl.read_csv(out, separator="\t", schema_overrides={"phecode": str})
        for col in ["phecode", "p_value", "neg_log_p_value", "beta", "cases", "controls"]:
            assert col in result.columns

    def test_p_values_valid(self, tmp_path):
        out = str(tmp_path / "aou_logit_pval.tsv")
        PheWAS(
            phecode_version="X",
            phecode_count_file_path=PHECODE_COUNT_FILE,
            cohort_file_path=COHORT_FILE,
            covariate_cols=COVARIATE_COLS,
            independent_variable_of_interest=INDEPENDENT_VAR,
            sex_at_birth_col=SEX_COL,
            min_cases=50,
            min_phecode_count=2,
            output_file_path=out,
        ).run()
        result = pl.read_csv(out, separator="\t", schema_overrides={"phecode": str})
        p_vals = result["p_value"].drop_nulls().to_list()
        assert all(0 <= p <= 1 for p in p_vals)


@pytest.mark.aou
class TestPheWASAouPlot:
    def test_manhattan_plot_saves(self, tmp_path):
        results_out = str(tmp_path / "results.tsv")
        PheWAS(
            phecode_version="X",
            phecode_count_file_path=PHECODE_COUNT_FILE,
            cohort_file_path=COHORT_FILE,
            covariate_cols=COVARIATE_COLS,
            independent_variable_of_interest=INDEPENDENT_VAR,
            sex_at_birth_col=SEX_COL,
            min_cases=50,
            min_phecode_count=2,
            output_file_path=results_out,
        ).run()

        plot_out = str(tmp_path / "manhattan.png")
        Plot(phewas_result_file_path=results_out, phecode_version="X").manhattan(
            save_plot=True, output_file_path=plot_out
        )
        assert os.path.exists(plot_out)
