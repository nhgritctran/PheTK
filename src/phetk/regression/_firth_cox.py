import numpy as np
import polars as pl

from phetk.regression._base import RegressionBackend, register


@register("firth_cox")
class FirthCoxBackend(RegressionBackend):
    """Firth penalized Cox proportional hazards regression backend using firthmodels."""
    requires_time_to_event = True

    def fit(
        self,
        regressors: pl.DataFrame,
        y: np.ndarray,
        analysis_var_cols: list[str],
        independent_variable_of_interest: str,
        **kwargs,
    ) -> dict[str, float | str] | None:
        from firthmodels import FirthCoxPH

        penalty_weight = kwargs.get("firth_penalty_weight", 0.5)
        max_iter = kwargs.get("firth_max_iter") or 50
        use_lrt = kwargs.get("firth_use_lrt", True)
        cox_stratification_col = kwargs.get("cox_stratification_col")
        verbose = kwargs.get("verbose", False)

        # Build feature cols (exclude "observed_time" and stratification col from X)
        feature_cols = [c for c in analysis_var_cols if c != "observed_time"]
        if cox_stratification_col and cox_stratification_col in feature_cols:
            feature_cols.remove(cox_stratification_col)

        X = regressors[feature_cols].to_numpy()
        var_index = feature_cols.index(independent_variable_of_interest)
        time = regressors["observed_time"].to_numpy().astype(float)

        # firthmodels has no missing="drop". Drop incomplete rows across X, time, and y
        # jointly, before casting event (NaN.astype(bool) would become True).
        valid = np.isfinite(X).all(axis=1) & np.isfinite(time) & np.isfinite(y)
        if not valid.all():
            X = X[valid]
            time = time[valid]
            y = y[valid]
        event = y.astype(bool)

        # Note: firthmodels FirthCoxPH does not support stratification natively
        stratified_by = "None"
        if cox_stratification_col and cox_stratification_col in regressors.columns:
            stratified_by = f"{cox_stratification_col} (not applied - unsupported by Firth Cox)"

        model = FirthCoxPH(max_iter=max_iter, penalty_weight=penalty_weight)
        try:
            model.fit(X, (event, time))
        except Exception as err:
            if verbose:
                print(f"FirthCoxBackend: {err}")
            return None

        log_hazard_ratio = float(model.coef_[var_index])

        # use_lrt=True  → LRT p, lrt_bse_ SE, profile-likelihood CI
        # use_lrt=False → Wald p, bse_ SE, Wald CI
        lrt_ok = False
        if use_lrt:
            try:
                model.lrt(var_index)
                p_value = float(model.lrt_pvalues_[var_index])
                standard_error = float(model.lrt_bse_[var_index])
                lrt_ok = True
            except Exception:
                pass

        if not lrt_ok:
            p_value = float(model.pvalues_[var_index])
            standard_error = float(model.bse_[var_index])

        if lrt_ok:
            try:
                ci = model.conf_int(alpha=0.05, method="pl", features=[var_index])
            except Exception:
                ci = model.conf_int(alpha=0.05, method="wald")
        else:
            ci = model.conf_int(alpha=0.05, method="wald")
        hazard_ratio = np.exp(log_hazard_ratio)
        hazard_ratio_low = np.exp(float(ci[var_index, 0]))
        hazard_ratio_high = np.exp(float(ci[var_index, 1]))

        # Concordance index
        try:
            concordance_index = model.score(X, (event, time))
        except Exception:
            concordance_index = np.nan

        convergence = "Converged" if model.converged_ else "Not converged"

        return {
            "p_value": p_value,
            "neg_log_p_value": -np.log10(p_value) if p_value > 0 else np.inf,
            "standard_error": standard_error,
            "hazard_ratio": hazard_ratio,
            "hazard_ratio_low": hazard_ratio_low,
            "hazard_ratio_high": hazard_ratio_high,
            "log_hazard_ratio": log_hazard_ratio,
            "concordance_index": concordance_index,
            "stratified_by": stratified_by,
            "convergence": convergence,
        }
