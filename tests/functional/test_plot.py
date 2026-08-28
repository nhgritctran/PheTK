"""
Functional tests for plot.py — runs PheWAS first, then tests all plot types.
No mocking, no AoU required. Uses non-interactive matplotlib backend.
"""
import os
import polars as pl
import pytest
import matplotlib
matplotlib.use("Agg")  # must be before any other matplotlib import

from phetk._utils import generate_mock_phewas_data
from phetk.phewas import PheWAS
from phetk.plot import Plot


@pytest.fixture(scope="module")
def phewas_result(tmp_path_factory):
    """Run PheWAS once and return the result file path."""
    tmp = tmp_path_factory.mktemp("plot_test")
    orig = os.getcwd()
    os.chdir(str(tmp))
    try:
        generate_mock_phewas_data(cohort_size=400, phecode="GE_979.2")
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


# ---------------------------------------------------------------------------
# Miami plot
# ---------------------------------------------------------------------------

class TestMiamiPlot:
    def test_saves_png(self, phewas_result, tmp_path):
        path, _ = phewas_result
        out = str(tmp_path / "miami.png")
        Plot(phewas_result_file_path=path, phecode_version="X").miami(
            save_plot=True, output_file_path=out
        )
        assert os.path.exists(out)

    def test_saves_pdf(self, phewas_result, tmp_path):
        path, _ = phewas_result
        out = str(tmp_path / "miami.pdf")
        Plot(phewas_result_file_path=path, phecode_version="X").miami(
            save_plot=True, output_file_path=out
        )
        assert os.path.exists(out)

    def test_auto_filename_generated(self, phewas_result, tmp_path, monkeypatch):
        path, _ = phewas_result
        monkeypatch.chdir(tmp_path)
        Plot(phewas_result_file_path=path, phecode_version="X").miami(save_plot=True)
        files = list(tmp_path.iterdir())
        assert any("miami" in f.name for f in files)

    def test_hide_non_significant(self, phewas_result, tmp_path):
        path, _ = phewas_result
        out = str(tmp_path / "miami_sig.png")
        Plot(phewas_result_file_path=path, phecode_version="X").miami(
            hide_non_significant=True, save_plot=True, output_file_path=out
        )
        assert os.path.exists(out)

    def test_diamond_capped_marker_style(self, phewas_result, tmp_path):
        path, _ = phewas_result
        out = str(tmp_path / "miami_diamond.png")
        Plot(phewas_result_file_path=path, phecode_version="X").miami(
            capped_marker_style="diamond", save_plot=True, output_file_path=out
        )
        assert os.path.exists(out)

    def test_custom_marker_alpha(self, phewas_result, tmp_path):
        path, _ = phewas_result
        out = str(tmp_path / "miami_alpha.png")
        Plot(phewas_result_file_path=path, phecode_version="X").miami(
            positive_marker_alpha=0.5, negative_marker_alpha=0.3,
            save_plot=True, output_file_path=out
        )
        assert os.path.exists(out)


# ---------------------------------------------------------------------------
# "converged" column backward compatibility
# ---------------------------------------------------------------------------

class TestConvergedBackwardCompatibility:
    """
    Plot must read the "converged" column as written by any PheTK version.

    Firth backends wrote "Converged"/"Not converged" up to v0.3.4, statsmodels logit
    wrote "True"/"False", and both write real booleans from v0.3.5 on.
    """

    # (true_token, false_token, description)
    FORMATS = [
        ("true", "false", "v0.3.5+ boolean"),
        ("True", "False", "statsmodels logit <= v0.3.4"),
        ("Converged", "Not converged", "Firth backends <= v0.3.4"),
    ]

    @staticmethod
    def _rewrite_converged(src_path, dst_path, true_token, false_token, n_non_converged=0):
        """Rewrite the converged column of a real result file using the given tokens."""
        df = pl.read_csv(src_path, separator="\t", schema_overrides={"phecode": str})
        tokens = [false_token] * n_non_converged + [true_token] * (len(df) - n_non_converged)
        df.with_columns(pl.Series("converged", tokens)).write_csv(dst_path, separator="\t")

    @pytest.mark.parametrize("true_token,false_token,description", FORMATS)
    def test_all_historical_formats_load(self, phewas_result, tmp_path,
                                         true_token, false_token, description):
        src, _ = phewas_result
        dst = str(tmp_path / "results.tsv")
        self._rewrite_converged(src, dst, true_token, false_token)

        plot = Plot(phewas_result_file_path=dst, phecode_version="X")

        assert plot.phewas_result.schema["converged"] == pl.Boolean, description
        assert plot.phewas_result["converged"].all(), description
        assert len(plot.phewas_result) > 0, description

    @pytest.mark.parametrize("true_token,false_token,description", FORMATS)
    def test_non_converged_rows_filtered(self, phewas_result, tmp_path,
                                         true_token, false_token, description):
        src, _ = phewas_result
        dst = str(tmp_path / "results_mixed.tsv")
        self._rewrite_converged(src, dst, true_token, false_token, n_non_converged=2)

        kept_all = Plot(phewas_result_file_path=dst, phecode_version="X",
                        converged_only=False).phewas_result
        kept_converged = Plot(phewas_result_file_path=dst, phecode_version="X",
                              converged_only=True).phewas_result

        assert len(kept_converged) < len(kept_all), description
        assert kept_converged["converged"].all(), description
        assert not kept_all["converged"].all(), description

    def test_unrecognized_value_becomes_null_and_is_filtered(self, phewas_result, tmp_path):
        src, _ = phewas_result
        dst = str(tmp_path / "results_unknown.tsv")
        self._rewrite_converged(src, dst, "true", "something unexpected", n_non_converged=2)

        kept_all = Plot(phewas_result_file_path=dst, phecode_version="X",
                        converged_only=False).phewas_result
        kept_converged = Plot(phewas_result_file_path=dst, phecode_version="X",
                              converged_only=True).phewas_result

        assert kept_all["converged"].null_count() == 2
        assert len(kept_converged) == len(kept_all) - 2

    def test_missing_converged_column_is_tolerated(self, phewas_result, tmp_path):
        src, _ = phewas_result
        dst = str(tmp_path / "results_no_converged.tsv")
        pl.read_csv(src, separator="\t", schema_overrides={"phecode": str}) \
            .drop("converged").write_csv(dst, separator="\t")

        plot = Plot(phewas_result_file_path=dst, phecode_version="X", converged_only=True)

        assert "converged" not in plot.phewas_result.columns
        assert len(plot.phewas_result) > 0
