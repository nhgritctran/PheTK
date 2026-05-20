from io import StringIO

import numpy as np
import pandas as pd
import polars as pl
import statsmodels
import statsmodels.api as sm

from phetk.regression._base import RegressionBackend, register


@register("logit")
class LogitBackend(RegressionBackend):
    """Standard logistic regression backend using statsmodels."""
    requires_time_to_event = False

    def fit(
        self,
        regressors: pl.DataFrame,
        y: np.ndarray,
        analysis_var_cols: list[str],
        independent_variable_of_interest: str,
        **kwargs,
    ) -> dict[str, float | str] | None:
        verbose = kwargs.get("verbose", False)
        suppress_warnings = kwargs.get("suppress_warnings", True)

        X_df = regressors[analysis_var_cols]
        var_index = X_df.columns.index(independent_variable_of_interest)
        X = X_df.to_numpy()
        X = sm.tools.add_constant(X, prepend=False)
        logit = sm.Logit(y, X, missing="drop")

        try:
            result = logit.fit(disp=False)
        except (np.linalg.linalg.LinAlgError, statsmodels.tools.sm_exceptions.PerfectSeparationError) as err:
            if "Singular matrix" in str(err) or "Perfect separation" in str(err):
                if verbose:
                    print(f"LogitBackend: {err}")
            else:
                raise
            return None

        return self._extract_results(result, var_index)

    @staticmethod
    def _extract_results(result, var_of_interest_index: int) -> dict[str, float | str]:
        results_as_html = result.summary().tables[0].as_html()
        converged = pd.read_html(StringIO(results_as_html))[0].iloc[5, 1]
        results_as_html = result.summary().tables[1].as_html()
        res = pd.read_html(StringIO(results_as_html), header=0, index_col=0)[0]

        p_value = result.pvalues[var_of_interest_index]
        neg_log_p_value = -np.log10(p_value)
        standard_error = res.iloc[var_of_interest_index]['std err']
        beta = result.params[var_of_interest_index]
        conf_int_1 = res.iloc[var_of_interest_index]['[0.025']
        conf_int_2 = res.iloc[var_of_interest_index]['0.975]']
        odds_ratio = np.exp(beta)
        log10_odds_ratio = np.log10(odds_ratio)

        return {
            "p_value": p_value,
            "neg_log_p_value": neg_log_p_value,
            "standard_error": standard_error,
            "beta": beta,
            "conf_int_1": conf_int_1,
            "conf_int_2": conf_int_2,
            "odds_ratio": odds_ratio,
            "log10_odds_ratio": log10_odds_ratio,
            "converged": converged,
        }
