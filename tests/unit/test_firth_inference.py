"""
Tests for Firth backend inference coherence: LRT vs Wald SE/CI and NA handling.
"""
import numpy as np
import polars as pl
import pytest
from scipy.stats import chi2

from phetk.regression import get_backend


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_quasi_separated_data(n=300, seed=42):
    """Generate quasi-separated logistic data (rare binary exposure, most carriers are cases).

    Returns:
        Tuple of (regressors, y, analysis_var_cols, var_index).
    """
    rng = np.random.default_rng(seed)
    # rare binary exposure (~10%)
    x_interest = (rng.random(n) < 0.10).astype(float)
    x1 = rng.standard_normal(n)
    # Most carriers are cases: strong positive effect
    logit = 3.0 * x_interest + 0.5 * x1
    prob = 1 / (1 + np.exp(-logit))
    y = rng.binomial(1, prob).astype(float)

    regressors = pl.DataFrame({
        "x_interest": x_interest,
        "x1": x1,
    })
    analysis_var_cols = ["x_interest", "x1"]
    var_index = 0  # x_interest
    return regressors, y, analysis_var_cols, var_index


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFirthLRT:
    """Verify that use_lrt=True yields lrt_bse_ (not bse_) and consistent p/SE."""

    def test_lrt_se_is_lrt_bse(self):
        from firthmodels import FirthLogisticRegression

        regressors, y, cols, var_index = _make_quasi_separated_data()
        backend = get_backend("firth_logit")

        # Fit via backend (use_lrt=True by default)
        result = backend.fit(
            regressors=regressors, y=y,
            analysis_var_cols=cols,
            independent_variable_of_interest="x_interest",
        )
        assert result is not None

        # Fit model directly to get ground truth lrt_bse_
        X = regressors[cols].to_numpy()
        model = FirthLogisticRegression(fit_intercept=True, max_iter=25, penalty_weight=0.5)
        model.fit(X, y)
        model.lrt(var_index)
        expected_se = float(model.lrt_bse_[var_index])

        assert result["standard_error"] == expected_se

        # Consistency: (beta/SE)^2 ≈ chi2.ppf(1 - p, 1) within rtol=1e-2
        beta = result["beta"]
        se = result["standard_error"]
        p = result["p_value"]
        if p < 1.0 and se > 0:
            wald_chi2 = (beta / se) ** 2
            expected_chi2 = chi2.ppf(1 - p, 1)
            np.testing.assert_allclose(wald_chi2, expected_chi2, rtol=1e-2)


class TestFirthWald:
    """Verify that use_lrt=False yields bse_ and Wald CI."""

    def test_wald_se_is_bse(self):
        from firthmodels import FirthLogisticRegression

        regressors, y, cols, var_index = _make_quasi_separated_data()
        backend = get_backend("firth_logit")

        result = backend.fit(
            regressors=regressors, y=y,
            analysis_var_cols=cols,
            independent_variable_of_interest="x_interest",
            firth_use_lrt=False,
        )
        assert result is not None

        # Ground truth bse_ and Wald CI
        X = regressors[cols].to_numpy()
        model = FirthLogisticRegression(fit_intercept=True, max_iter=25, penalty_weight=0.5)
        model.fit(X, y)
        expected_se = float(model.bse_[var_index])
        expected_ci = model.conf_int(alpha=0.05, method="wald")

        assert result["standard_error"] == expected_se
        np.testing.assert_allclose(result["conf_int_1"], float(expected_ci[var_index, 0]))
        np.testing.assert_allclose(result["conf_int_2"], float(expected_ci[var_index, 1]))


class TestPLvsWaldCI:
    """Verify that PL CI (LRT mode) differs from Wald CI."""

    def test_pl_ci_vs_wald_ci_differ(self):
        regressors, y, cols, var_index = _make_quasi_separated_data()
        backend = get_backend("firth_logit")

        lrt_result = backend.fit(
            regressors=regressors, y=y,
            analysis_var_cols=cols,
            independent_variable_of_interest="x_interest",
            firth_use_lrt=True,
        )
        wald_result = backend.fit(
            regressors=regressors, y=y,
            analysis_var_cols=cols,
            independent_variable_of_interest="x_interest",
            firth_use_lrt=False,
        )
        assert lrt_result is not None
        assert wald_result is not None

        # Both should return finite values
        for key in ("conf_int_1", "conf_int_2"):
            assert np.isfinite(lrt_result[key])
            assert np.isfinite(wald_result[key])

        # The two intervals should differ (PL != Wald for quasi-separated data)
        assert (
            lrt_result["conf_int_1"] != wald_result["conf_int_1"]
            or lrt_result["conf_int_2"] != wald_result["conf_int_2"]
        )

        # LRT CI should match model's PL CI
        from firthmodels import FirthLogisticRegression
        X = regressors[cols].to_numpy()
        model = FirthLogisticRegression(fit_intercept=True, max_iter=25, penalty_weight=0.5)
        model.fit(X, y)
        pl_ci = model.conf_int(alpha=0.05, method="pl", features=[var_index])
        np.testing.assert_allclose(lrt_result["conf_int_1"], float(pl_ci[var_index, 0]))
        np.testing.assert_allclose(lrt_result["conf_int_2"], float(pl_ci[var_index, 1]))


class TestNAHandling:
    """Verify that polars nulls (NaN in numpy) are dropped before fitting."""

    def test_na_handling(self):
        regressors, y, cols, var_index = _make_quasi_separated_data()
        backend = get_backend("firth_logit")

        # Inject a null into one covariate in one row
        dirty_regressors = regressors.with_columns(
            pl.when(pl.int_range(pl.len()) == 5)
            .then(None)
            .otherwise(pl.col("x1"))
            .alias("x1")
        )

        # Fit on dirty data (should not crash)
        dirty_result = backend.fit(
            regressors=dirty_regressors, y=y,
            analysis_var_cols=cols,
            independent_variable_of_interest="x_interest",
        )
        assert dirty_result is not None

        # Fit on manually cleaned data (row 5 removed)
        mask = np.ones(len(y), dtype=bool)
        mask[5] = False
        clean_regressors = regressors.filter(pl.Series(mask))
        clean_y = y[mask]
        clean_result = backend.fit(
            regressors=clean_regressors, y=clean_y,
            analysis_var_cols=cols,
            independent_variable_of_interest="x_interest",
        )
        assert clean_result is not None

        # Results should match
        np.testing.assert_allclose(dirty_result["beta"], clean_result["beta"])
        np.testing.assert_allclose(dirty_result["p_value"], clean_result["p_value"])
