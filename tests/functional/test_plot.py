"""
Functional tests for plot.py — runs PheWAS first, then tests all plot types.
No mocking, no AoU required. Uses non-interactive matplotlib backend.
"""
import os
import pytest
import matplotlib
matplotlib.use("Agg")  # must be before any other matplotlib import

from phetk.demo import generate_examples
from phetk.phewas import PheWAS
from phetk.plot import Plot


@pytest.fixture(scope="module")
def phewas_result(tmp_path_factory):
    """Run PheWAS once and return the result file path."""
    tmp = tmp_path_factory.mktemp("plot_test")
    orig = os.getcwd()
    os.chdir(str(tmp))
    try:
        generate_examples(cohort_size=400, phecode="GE_979.2")
    finally:
        os.chdir(orig)

    out = str(tmp / "phewas_results.tsv")
    PheWAS(
        phecode_version="X",
        phecode_count_file_path=str(tmp / "example_phecode_counts.tsv"),
        cohort_file_path=str(tmp / "example_cohort.tsv"),
        covariate_cols=["age", "sex", "pc1", "pc2", "pc3"],
        independent_variable_of_interest="independent_variable_of_interest",
        sex_at_birth_col="sex",
        min_cases=5,
        min_phecode_count=2,
        output_file_path=out,
    ).run()
    return out, tmp


# ---------------------------------------------------------------------------
# Plot initialisation
# ---------------------------------------------------------------------------

class TestPlotInit:
    def test_loads_results(self, phewas_result):
        path, _ = phewas_result
        plot = Plot(phewas_result_file_path=path, phecode_version="X")
        assert plot.phewas_result is not None
        assert len(plot.phewas_result) > 0

    def test_bonferroni_auto_calculated(self, phewas_result):
        path, _ = phewas_result
        plot = Plot(phewas_result_file_path=path, phecode_version="X")
        assert plot.bonferroni is not None
        assert plot.bonferroni > 0

    def test_custom_bonferroni_used(self, phewas_result):
        path, _ = phewas_result
        plot = Plot(phewas_result_file_path=path, phecode_version="X", bonferroni=0.05)
        assert plot.bonferroni == 0.05

    def test_default_color_palette(self, phewas_result):
        path, _ = phewas_result
        plot = Plot(phewas_result_file_path=path, phecode_version="X")
        assert plot is not None

    def test_colorblind_palette(self, phewas_result):
        path, _ = phewas_result
        plot = Plot(phewas_result_file_path=path, phecode_version="X", color_palette="colorblind")
        assert plot is not None

    def test_rainbow_palette(self, phewas_result):
        path, _ = phewas_result
        plot = Plot(phewas_result_file_path=path, phecode_version="X", color_palette="rainbow")
        assert plot is not None


# ---------------------------------------------------------------------------
# Manhattan plot
# ---------------------------------------------------------------------------

class TestManhattanPlot:
    def test_saves_png(self, phewas_result, tmp_path):
        path, _ = phewas_result
        out = str(tmp_path / "manhattan.png")
        Plot(phewas_result_file_path=path, phecode_version="X").manhattan(
            save_plot=True, output_file_path=out
        )
        assert os.path.exists(out)

    def test_saves_pdf(self, phewas_result, tmp_path):
        path, _ = phewas_result
        out = str(tmp_path / "manhattan.pdf")
        Plot(phewas_result_file_path=path, phecode_version="X").manhattan(
            save_plot=True, output_file_path=out
        )
        assert os.path.exists(out)

    def test_auto_filename_generated(self, phewas_result, tmp_path, monkeypatch):
        path, _ = phewas_result
        monkeypatch.chdir(tmp_path)
        Plot(phewas_result_file_path=path, phecode_version="X").manhattan(save_plot=True)
        # At least one file starting with "manhattan_" should exist
        files = list(tmp_path.iterdir())
        assert any("manhattan" in f.name for f in files)


# ---------------------------------------------------------------------------
# Volcano plot
# ---------------------------------------------------------------------------

class TestVolcanoPlot:
    def test_saves_png(self, phewas_result, tmp_path):
        path, _ = phewas_result
        out = str(tmp_path / "volcano.png")
        Plot(phewas_result_file_path=path, phecode_version="X").volcano(
            save_plot=True, output_file_path=out
        )
        assert os.path.exists(out)


# ---------------------------------------------------------------------------
# Forest plot
# ---------------------------------------------------------------------------

class TestForestPlot:
    def test_saves_png(self, phewas_result, tmp_path):
        path, _ = phewas_result
        out = str(tmp_path / "forest.png")
        Plot(phewas_result_file_path=path, phecode_version="X").forest(
            save_plot=True, output_file_path=out
        )
        assert os.path.exists(out)

    def test_saves_pdf(self, phewas_result, tmp_path):
        path, _ = phewas_result
        out = str(tmp_path / "forest.pdf")
        Plot(phewas_result_file_path=path, phecode_version="X").forest(
            save_plot=True, output_file_path=out
        )
        assert os.path.exists(out)
