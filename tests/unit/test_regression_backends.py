"""
Unit tests for the regression backend registry and individual backends.
"""
import numpy as np
import polars as pl
import pytest

from phetk.regression import available_methods, get_backend
from phetk.regression._base import _REGISTRY


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_all_methods_registered(self):
        methods = available_methods()
        assert "logit" in methods
        assert "cox" in methods
        assert "firth_logit" in methods
        assert "firth_cox" in methods

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown method"):
            get_backend("nonexistent")

    def test_get_backend_returns_instance(self):
        for method in available_methods():
            backend = get_backend(method)
            assert backend is not None

    def test_time_to_event_flags(self):
        assert get_backend("logit").requires_time_to_event is False
        assert get_backend("cox").requires_time_to_event is True
        assert get_backend("firth_logit").requires_time_to_event is False
        assert get_backend("firth_cox").requires_time_to_event is True


# ---------------------------------------------------------------------------
# Helpers to build synthetic data
# ---------------------------------------------------------------------------

def _make_logistic_data(n=200, seed=42):
    """Generate linearly separable logistic data."""
    rng = np.random.default_rng(seed)
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    x_interest = rng.integers(0, 2, n).astype(float)
    prob = 1 / (1 + np.exp(-(0.5 * x_interest + 0.3 * x1 - 0.2 * x2)))
    y = rng.binomial(1, prob).astype(float)

    regressors = pl.DataFrame({
        "x_interest": x_interest,
        "x1": x1,
        "x2": x2,
    })
    analysis_var_cols = ["x_interest", "x1", "x2"]
    return regressors, y, analysis_var_cols


def _make_survival_data(n=200, seed=42):
    """Generate synthetic survival data.

    The regressors DataFrame includes a "y" column because CoxPHFitter.fit()
    expects event_col="y" to be present in the DataFrame.
    """
    rng = np.random.default_rng(seed)
    x1 = rng.standard_normal(n)
    x_interest = rng.integers(0, 2, n).astype(float)
    observed_time = rng.exponential(5.0, n)
    observed_time = np.clip(observed_time, 0.1, None)
    prob = 1 / (1 + np.exp(-(0.3 * x_interest + 0.2 * x1)))
    y = rng.binomial(1, prob).astype(float)

    regressors = pl.DataFrame({
        "x_interest": x_interest,
        "x1": x1,
        "observed_time": observed_time,
        "y": y,
    })
    analysis_var_cols = ["x_interest", "x1", "observed_time"]
    return regressors, y, analysis_var_cols


# ---------------------------------------------------------------------------
# LogitBackend
# ---------------------------------------------------------------------------

class TestLogitBackend:
    def test_fit_returns_expected_keys(self):
        regressors, y, cols = _make_logistic_data()
        backend = get_backend("logit")
        result = backend.fit(
            regressors=regressors, y=y,
            analysis_var_cols=cols,
            independent_variable_of_interest="x_interest",
        )
        assert result is not None
        expected_keys = {
            "p_value", "neg_log_p_value", "standard_error", "beta",
            "conf_int_1", "conf_int_2", "odds_ratio", "log10_odds_ratio", "converged",
        }
        assert set(result.keys()) == expected_keys

    def test_p_value_between_0_and_1(self):
        regressors, y, cols = _make_logistic_data()
        backend = get_backend("logit")
        result = backend.fit(
            regressors=regressors, y=y,
            analysis_var_cols=cols,
            independent_variable_of_interest="x_interest",
        )
        assert 0 < result["p_value"] <= 1


# ---------------------------------------------------------------------------
# CoxBackend
# ---------------------------------------------------------------------------

class TestCoxBackend:
    def test_fit_returns_expected_keys(self):
        regressors, y, cols = _make_survival_data()
        backend = get_backend("cox")
        result = backend.fit(
            regressors=regressors, y=y,
            analysis_var_cols=cols,
            independent_variable_of_interest="x_interest",
        )
        assert result is not None
        expected_keys = {
            "p_value", "neg_log_p_value", "standard_error",
            "hazard_ratio", "hazard_ratio_low", "hazard_ratio_high",
            "log_hazard_ratio", "concordance_index", "stratified_by", "convergence",
        }
        assert set(result.keys()) == expected_keys

    def test_p_value_between_0_and_1(self):
        regressors, y, cols = _make_survival_data()
        backend = get_backend("cox")
        result = backend.fit(
            regressors=regressors, y=y,
            analysis_var_cols=cols,
            independent_variable_of_interest="x_interest",
        )
        assert 0 < result["p_value"] <= 1


# ---------------------------------------------------------------------------
# FirthLogitBackend
# ---------------------------------------------------------------------------

class TestFirthLogitBackend:
    def test_fit_returns_expected_keys(self):
        regressors, y, cols = _make_logistic_data()
        backend = get_backend("firth_logit")
        result = backend.fit(
            regressors=regressors, y=y,
            analysis_var_cols=cols,
            independent_variable_of_interest="x_interest",
        )
        assert result is not None
        expected_keys = {
            "p_value", "neg_log_p_value", "standard_error", "beta",
            "conf_int_1", "conf_int_2", "odds_ratio", "log10_odds_ratio", "converged",
        }
        assert set(result.keys()) == expected_keys

    def test_p_value_between_0_and_1(self):
        regressors, y, cols = _make_logistic_data()
        backend = get_backend("firth_logit")
        result = backend.fit(
            regressors=regressors, y=y,
            analysis_var_cols=cols,
            independent_variable_of_interest="x_interest",
        )
        assert 0 < result["p_value"] <= 1

    def test_output_columns_match_logit(self):
        regressors, y, cols = _make_logistic_data()
        logit_result = get_backend("logit").fit(
            regressors=regressors, y=y,
            analysis_var_cols=cols,
            independent_variable_of_interest="x_interest",
        )
        firth_result = get_backend("firth_logit").fit(
            regressors=regressors, y=y,
            analysis_var_cols=cols,
            independent_variable_of_interest="x_interest",
        )
        assert set(logit_result.keys()) == set(firth_result.keys())

    def test_wald_pvalues_option(self):
        regressors, y, cols = _make_logistic_data()
        backend = get_backend("firth_logit")
        result = backend.fit(
            regressors=regressors, y=y,
            analysis_var_cols=cols,
            independent_variable_of_interest="x_interest",
            firth_use_lrt=False,
        )
        assert result is not None
        assert 0 < result["p_value"] <= 1


# ---------------------------------------------------------------------------
# FirthCoxBackend
# ---------------------------------------------------------------------------

class TestFirthCoxBackend:
    def test_fit_returns_expected_keys(self):
        regressors, y, cols = _make_survival_data()
        backend = get_backend("firth_cox")
        result = backend.fit(
            regressors=regressors, y=y,
            analysis_var_cols=cols,
            independent_variable_of_interest="x_interest",
        )
        assert result is not None
        expected_keys = {
            "p_value", "neg_log_p_value", "standard_error",
            "hazard_ratio", "hazard_ratio_low", "hazard_ratio_high",
            "log_hazard_ratio", "concordance_index", "stratified_by", "convergence",
        }
        assert set(result.keys()) == expected_keys

    def test_p_value_between_0_and_1(self):
        regressors, y, cols = _make_survival_data()
        backend = get_backend("firth_cox")
        result = backend.fit(
            regressors=regressors, y=y,
            analysis_var_cols=cols,
            independent_variable_of_interest="x_interest",
        )
        assert 0 < result["p_value"] <= 1

    def test_output_columns_match_cox(self):
        regressors, y, cols = _make_survival_data()
        cox_result = get_backend("cox").fit(
            regressors=regressors, y=y,
            analysis_var_cols=cols,
            independent_variable_of_interest="x_interest",
        )
        firth_result = get_backend("firth_cox").fit(
            regressors=regressors, y=y,
            analysis_var_cols=cols,
            independent_variable_of_interest="x_interest",
        )
        assert set(cox_result.keys()) == set(firth_result.keys())

    def test_wald_pvalues_option(self):
        regressors, y, cols = _make_survival_data()
        backend = get_backend("firth_cox")
        result = backend.fit(
            regressors=regressors, y=y,
            analysis_var_cols=cols,
            independent_variable_of_interest="x_interest",
            firth_use_lrt=False,
        )
        assert result is not None
        assert 0 < result["p_value"] <= 1
