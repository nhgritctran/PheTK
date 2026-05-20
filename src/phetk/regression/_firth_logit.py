import numpy as np
import polars as pl

from phetk.regression._base import RegressionBackend, register


@register("firth_logit")
class FirthLogitBackend(RegressionBackend):
    """Firth penalized logistic regression backend using firthmodels."""
    requires_time_to_event = False

    def fit(
        self,
        regressors: pl.DataFrame,
        y: np.ndarray,
        analysis_var_cols: list[str],
        independent_variable_of_interest: str,
        **kwargs,
    ) -> dict[str, float | str] | None:
        from firthmodels import FirthLogisticRegression

        penalty_weight = kwargs.get("firth_penalty_weight", 0.5)
        max_iter = kwargs.get("firth_max_iter") or 25
        use_lrt = kwargs.get("firth_use_lrt", True)
        verbose = kwargs.get("verbose", False)

        X = regressors[analysis_var_cols].to_numpy()
        var_index = analysis_var_cols.index(independent_variable_of_interest)

        model = FirthLogisticRegression(
            fit_intercept=True, max_iter=max_iter, penalty_weight=penalty_weight
        )
        try:
            model.fit(X, y)
        except Exception as err:
            if verbose:
                print(f"FirthLogitBackend: {err}")
            return None

        # LRT p-values by default (more accurate for small samples)
        if use_lrt:
            try:
                model.lrt()
                p_value = float(model.lrt_pvalues_[var_index])
            except Exception:
                p_value = float(model.pvalues_[var_index])
        else:
            p_value = float(model.pvalues_[var_index])

        beta = float(model.coef_[var_index])
        standard_error = float(model.bse_[var_index])
        ci = model.conf_int(alpha=0.05, method="wald")
        conf_int_1 = float(ci[var_index, 0])
        conf_int_2 = float(ci[var_index, 1])
        odds_ratio = np.exp(beta)
        log10_odds_ratio = np.log10(odds_ratio) if odds_ratio > 0 else -np.inf
        converged = "Converged" if model.converged_ else "Not converged"

        return {
            "p_value": p_value,
            "neg_log_p_value": -np.log10(p_value) if p_value > 0 else np.inf,
            "standard_error": standard_error,
            "beta": beta,
            "conf_int_1": conf_int_1,
            "conf_int_2": conf_int_2,
            "odds_ratio": odds_ratio,
            "log10_odds_ratio": log10_odds_ratio,
            "converged": converged,
        }
