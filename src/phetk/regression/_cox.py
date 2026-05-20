import numpy as np
import polars as pl
import warnings

from lifelines import CoxPHFitter, utils as u

from phetk.regression._base import RegressionBackend, register


@register("cox")
class CoxBackend(RegressionBackend):
    """Standard Cox proportional hazards regression backend using lifelines."""
    requires_time_to_event = True

    def fit(
        self,
        regressors: pl.DataFrame,
        y: np.ndarray,
        analysis_var_cols: list[str],
        independent_variable_of_interest: str,
        **kwargs,
    ) -> dict[str, float | str] | None:
        cox_stratification_col = kwargs.get("cox_stratification_col")
        cox_fallback_step_size = kwargs.get("cox_fallback_step_size", 0.1)
        verbose = kwargs.get("verbose", False)

        # For cox regression, warnings are always on to catch convergence status
        warnings.simplefilter("always")

        strata = None
        stratified_by = "None"
        if cox_stratification_col in regressors.columns:
            strata = stratified_by = cox_stratification_col

        cox = CoxPHFitter()
        combined_warning = "Converged"
        try:
            captured_warnings = []
            with warnings.catch_warnings(record=True) as w:
                result = cox.fit(
                    df=regressors.to_pandas(use_pyarrow_extension_array=True),
                    event_col="y",
                    duration_col="observed_time",
                    strata=strata,
                )
            for warning in w:
                warning_message = str(warning.message)
                captured_warnings.append(warning_message)
            if captured_warnings:
                combined_warning = "\n".join(captured_warnings)
        except u.ConvergenceError:
            combined_warning = (
                f"Convergence error. step_size was lowered to "
                f"{cox_fallback_step_size} (default is 0.95)."
            )
            result = cox.fit(
                df=regressors.to_pandas(use_pyarrow_extension_array=True),
                event_col="y",
                duration_col="observed_time",
                strata=strata,
                fit_options={"step_size": cox_fallback_step_size},
            )
        except Exception as e:
            print("Exception:", e)
            return None

        return self._extract_results(
            result, independent_variable_of_interest, stratified_by, combined_warning
        )

    @staticmethod
    def _extract_results(
        result,
        independent_variable_of_interest: str,
        stratified_by: str,
        warning_message: str | None = None,
    ) -> dict[str, float | str]:
        result_df = result.summary

        p_value = result_df.loc[independent_variable_of_interest]["p"]
        neg_log_p_value = -np.log10(p_value)
        standard_error = result_df.loc[independent_variable_of_interest]["se(coef)"]
        hazard_ratio = result_df.loc[independent_variable_of_interest]["exp(coef)"]
        hazard_ratio_low = result_df.loc[independent_variable_of_interest]["exp(coef) lower 95%"]
        hazard_ratio_high = result_df.loc[independent_variable_of_interest]["exp(coef) upper 95%"]
        log_hazard_ratio = result_df.loc[independent_variable_of_interest]["coef"]

        concordance_index = result.concordance_index_

        return {
            "p_value": p_value,
            "neg_log_p_value": neg_log_p_value,
            "standard_error": standard_error,
            "hazard_ratio": hazard_ratio,
            "hazard_ratio_low": hazard_ratio_low,
            "hazard_ratio_high": hazard_ratio_high,
            "log_hazard_ratio": log_hazard_ratio,
            "concordance_index": concordance_index,
            "stratified_by": stratified_by,
            "convergence": warning_message,
        }
