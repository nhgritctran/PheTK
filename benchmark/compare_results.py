"""
Compare Firth vs non-Firth PheWAS results.

Reads paired TSV outputs from run_benchmark.py and produces:
  - A formatted text summary to stdout
  - 12 PNG plots in benchmark/comparison_plots/
"""

import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BENCHMARK_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(BENCHMARK_DIR, "comparison_plots")

PAIRS = [
    {
        "label": "Logistic: logit vs firth_logit",
        "tag": "logit",
        "std_file": "phewas_results_logit.tsv",
        "firth_file": "phewas_results_firth_logit.tsv",
        "effect_col": "beta",
        "ci_low": "conf_int_1",
        "ci_high": "conf_int_2",
        "convergence_col_std": "converged",
        "convergence_col_firth": "converged",
        "ci_on_hr_scale": False,
    },
    {
        "label": "Cox: cox vs firth_cox",
        "tag": "cox",
        "std_file": "phewas_results_cox.tsv",
        "firth_file": "phewas_results_firth_cox.tsv",
        "effect_col": "log_hazard_ratio",
        "ci_low": "hazard_ratio_low",
        "ci_high": "hazard_ratio_high",
        "convergence_col_std": "convergence",
        "convergence_col_firth": "convergence",
        "ci_on_hr_scale": True,
    },
]

# ---------------------------------------------------------------------------
# Helpers — stats (numpy-only, no scipy)
# ---------------------------------------------------------------------------


def _pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return np.nan
    xm, ym = x - x.mean(), y - y.mean()
    denom = np.sqrt((xm ** 2).sum() * (ym ** 2).sum())
    if denom == 0:
        return np.nan
    return float((xm * ym).sum() / denom)


def _spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return np.nan

    def _rank(arr: np.ndarray) -> np.ndarray:
        order = arr.argsort()
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, len(arr) + 1, dtype=float)
        # average ties
        for val in np.unique(arr):
            mask = arr == val
            ranks[mask] = ranks[mask].mean()
        return ranks

    return _pearson_r(_rank(x), _rank(y))


def _cohens_kappa(table: np.ndarray) -> float:
    """Cohen's kappa for a 2x2 contingency table."""
    n = table.sum()
    if n == 0:
        return np.nan
    p_o = (table[0, 0] + table[1, 1]) / n
    row_sums = table.sum(axis=1)
    col_sums = table.sum(axis=0)
    p_e = (row_sums[0] * col_sums[0] + row_sums[1] * col_sums[1]) / (n ** 2)
    if p_e == 1:
        return np.nan
    return float((p_o - p_e) / (1 - p_e))


def _mcnemar_chi2(table: np.ndarray) -> tuple[float, float]:
    """McNemar's test (chi-squared approximation) for a 2x2 table.

    Returns:
        (chi2_statistic, p_value) — p-value via chi-squared survival function
        approximated with the Wilson-Hilferty normal approximation.
    """
    b, c = float(table[0, 1]), float(table[1, 0])
    if b + c == 0:
        return 0.0, 1.0
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)  # continuity-corrected
    # chi-squared survival function (1 df) via normal approximation
    # For chi2 with 1 df, p = 2 * (1 - Phi(sqrt(chi2)))
    # Use complementary error function: erfc(x/sqrt(2))/2 = 1-Phi(x)
    # numpy doesn't have erfc, so use the series expansion for small p
    # Instead, use the Wilson-Hilferty approximation for chi2 -> z
    if chi2 == 0:
        return 0.0, 1.0
    z = np.sqrt(chi2)  # sqrt(chi2_1df) ~ N(0,1)
    # Approximate 1-Phi(z) using logistic approximation: 1/(1+exp(1.7*z))
    # More accurate: use the rational approximation
    p_value = _normal_survival(z) * 2  # two-tailed -> one-tailed for chi2(1)
    # chi2(1df) is one-tailed: P(X>chi2) = 2*P(Z>sqrt(chi2)) / 2 = P(Z>sqrt)
    p_value = _normal_survival(z) * 2  # survival of |Z|
    # Actually for chi2 with 1 df: p = P(chi2_1 > val) = P(|Z| > sqrt(val))
    # = 2 * P(Z > sqrt(val)) where Z ~ N(0,1)
    # But we want the upper tail of chi2, which equals 2*Phi_survival(sqrt(chi2))
    p_value = 2.0 * _normal_survival(z)
    return float(chi2), float(min(p_value, 1.0))


def _normal_survival(z: float) -> float:
    """Approximate P(Z > z) for z >= 0 using Abramowitz & Stegun 26.2.17."""
    if z < 0:
        return 1.0 - _normal_survival(-z)
    # constants
    b1, b2, b3, b4, b5 = 0.319381530, -0.356563782, 1.781477937, -1.882496350, 1.330274429
    p = 0.2316419
    t = 1.0 / (1.0 + p * z)
    pdf = np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
    return float(pdf * (b1 * t + b2 * t**2 + b3 * t**3 + b4 * t**4 + b5 * t**5))


def _paired_t_test(diffs: np.ndarray) -> tuple[float, float]:
    """One-sample t-test on diffs (H0: mean == 0).

    Returns:
        (t_statistic, p_value)
    """
    n = len(diffs)
    if n < 2:
        return np.nan, np.nan
    mean = diffs.mean()
    se = diffs.std(ddof=1) / np.sqrt(n)
    if se == 0:
        return np.nan, np.nan
    t_stat = mean / se
    # Approximate p-value using normal for large n
    p_value = 2.0 * _normal_survival(abs(t_stat))
    return float(t_stat), float(p_value)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _normalize_convergence(series: pl.Series, col_name: str) -> pl.Series:
    """Convert convergence column to boolean series.

    Standard logit: "True"/"False" (string from HTML parse)
    Firth logit: "Converged"/"Not converged"
    Standard cox: "Converged" or warning string
    Firth cox: "Converged"/"Not converged"
    """
    return (
        series.cast(pl.Utf8).str.to_lowercase()
        .str.starts_with("true").cast(pl.Boolean)
        | series.cast(pl.Utf8).str.to_lowercase()
        .str.starts_with("converged").cast(pl.Boolean)
    ).alias(col_name)


def load_pair(pair: dict) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame] | None:
    """Load and join a standard/firth pair of result TSVs.

    Returns:
        (std_df, firth_df, joined_df) with suffixed columns, or None if files
        are missing.
    """
    std_path = os.path.join(BENCHMARK_DIR, pair["std_file"])
    firth_path = os.path.join(BENCHMARK_DIR, pair["firth_file"])

    for path, name in [(std_path, "standard"), (firth_path, "firth")]:
        if not os.path.exists(path):
            print(f"WARNING: {name} file not found: {path} — skipping pair.")
            return None

    std_df = pl.read_csv(std_path, separator="\t", schema_overrides={"phecode": str})
    firth_df = pl.read_csv(firth_path, separator="\t", schema_overrides={"phecode": str})

    # normalize convergence to boolean
    conv_std = pair["convergence_col_std"]
    conv_firth = pair["convergence_col_firth"]
    std_df = std_df.with_columns(_normalize_convergence(std_df[conv_std], conv_std))
    firth_df = firth_df.with_columns(_normalize_convergence(firth_df[conv_firth], conv_firth))

    # rename convergence columns to a uniform name for joining
    std_df = std_df.rename({conv_std: "converged_bool"})
    firth_df = firth_df.rename({conv_firth: "converged_bool"})

    # suffix before join
    std_cols = {c: f"{c}_std" for c in std_df.columns if c != "phecode"}
    firth_cols = {c: f"{c}_firth" for c in firth_df.columns if c != "phecode"}
    std_renamed = std_df.rename(std_cols)
    firth_renamed = firth_df.rename(firth_cols)

    joined = std_renamed.join(firth_renamed, on="phecode", how="inner")
    return std_df, firth_df, joined


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_metrics(pair: dict, joined: pl.DataFrame) -> dict:
    """Compute all comparison metrics for a pair."""
    effect = pair["effect_col"]
    ci_low = pair["ci_low"]
    ci_high = pair["ci_high"]
    ci_hr = pair["ci_on_hr_scale"]

    results: dict = {}
    n_total = len(joined)
    results["n_matched"] = n_total

    # --- Convergence ---
    conv_std = joined["converged_bool_std"].to_numpy()
    conv_firth = joined["converged_bool_firth"].to_numpy()
    results["conv_std_count"] = int(conv_std.sum())
    results["conv_firth_count"] = int(conv_firth.sum())
    results["conv_std_rate"] = results["conv_std_count"] / max(n_total, 1)
    results["conv_firth_rate"] = results["conv_firth_count"] / max(n_total, 1)
    results["conv_std_only"] = int((conv_std & ~conv_firth).sum())
    results["conv_firth_only"] = int((~conv_std & conv_firth).sum())

    # --- Effect agreement ---
    e_std = joined[f"{effect}_std"].to_numpy().astype(float)
    e_firth = joined[f"{effect}_firth"].to_numpy().astype(float)
    valid_effect = np.isfinite(e_std) & np.isfinite(e_firth)
    e_std_v, e_firth_v = e_std[valid_effect], e_firth[valid_effect]

    results["effect_n"] = int(valid_effect.sum())
    results["effect_pearson"] = _pearson_r(e_std_v, e_firth_v)
    results["effect_spearman"] = _spearman_rho(e_std_v, e_firth_v)
    diff = np.abs(e_std_v - e_firth_v)
    results["effect_mad"] = float(diff.mean()) if len(diff) > 0 else np.nan
    results["effect_rmsd"] = float(np.sqrt((diff ** 2).mean())) if len(diff) > 0 else np.nan
    if len(diff) > 0:
        max_idx = diff.argmax()
        results["effect_max_diff"] = float(diff[max_idx])
        phecodes = joined.filter(pl.Series(valid_effect))["phecode"].to_list()
        results["effect_max_diff_phecode"] = phecodes[max_idx]
    else:
        results["effect_max_diff"] = np.nan
        results["effect_max_diff_phecode"] = "N/A"

    # store arrays for plots
    results["_e_std"] = e_std_v
    results["_e_firth"] = e_firth_v

    # --- P-value agreement ---
    p_std = joined["p_value_std"].to_numpy().astype(float)
    p_firth = joined["p_value_firth"].to_numpy().astype(float)

    # replace 0 with tiny
    tiny = np.finfo(float).tiny
    p_std = np.where(p_std == 0, tiny, p_std)
    p_firth = np.where(p_firth == 0, tiny, p_firth)

    valid_p = np.isfinite(p_std) & np.isfinite(p_firth) & (p_std > 0) & (p_firth > 0)
    p_std_v, p_firth_v = p_std[valid_p], p_firth[valid_p]

    nlp_std = -np.log10(p_std_v)
    nlp_firth = -np.log10(p_firth_v)
    # cap inf at 300
    nlp_std = np.minimum(nlp_std, 300.0)
    nlp_firth = np.minimum(nlp_firth, 300.0)

    results["pval_n"] = int(valid_p.sum())
    results["pval_pearson"] = _pearson_r(nlp_std, nlp_firth)
    results["pval_spearman"] = _spearman_rho(nlp_std, nlp_firth)

    # concordance tables at thresholds
    bonferroni_thresh = 0.05 / max(n_total, 1)
    thresholds = {"0.05": 0.05, "0.01": 0.01, "bonferroni": bonferroni_thresh}
    results["bonferroni_thresh"] = bonferroni_thresh
    results["concordance"] = {}
    for name, thresh in thresholds.items():
        sig_std = p_std_v < thresh
        sig_firth = p_firth_v < thresh
        table = np.array([
            [int((sig_std & sig_firth).sum()), int((sig_std & ~sig_firth).sum())],
            [int((~sig_std & sig_firth).sum()), int((~sig_std & ~sig_firth).sum())],
        ])
        kappa = _cohens_kappa(table)
        chi2, mcn_p = _mcnemar_chi2(table)
        results["concordance"][name] = {
            "table": table,
            "kappa": kappa,
            "mcnemar_chi2": chi2,
            "mcnemar_p": mcn_p,
        }

    results["_nlp_std"] = nlp_std
    results["_nlp_firth"] = nlp_firth
    results["_p_std_v"] = p_std_v
    results["_p_firth_v"] = p_firth_v

    # --- SE comparison ---
    se_std = joined["standard_error_std"].to_numpy().astype(float)
    se_firth = joined["standard_error_firth"].to_numpy().astype(float)
    valid_se = np.isfinite(se_std) & np.isfinite(se_firth) & (se_std > 0) & (se_firth > 0)
    se_std_v, se_firth_v = se_std[valid_se], se_firth[valid_se]
    se_ratio = se_firth_v / se_std_v

    results["se_n"] = int(valid_se.sum())
    results["se_zero_count"] = int(((se_std == 0) | (se_firth == 0)).sum())
    results["se_mean_ratio"] = float(se_ratio.mean()) if len(se_ratio) > 0 else np.nan
    results["se_median_ratio"] = float(np.median(se_ratio)) if len(se_ratio) > 0 else np.nan
    results["se_firth_larger_frac"] = float((se_ratio > 1.0).mean()) if len(se_ratio) > 0 else np.nan
    se_diff = se_firth_v - se_std_v
    t_stat, t_p = _paired_t_test(se_diff)
    results["se_paired_t"] = t_stat
    results["se_paired_t_p"] = t_p
    results["_se_ratio"] = se_ratio

    # --- CI overlap ---
    ci_low_std = joined[f"{ci_low}_std"].to_numpy().astype(float)
    ci_high_std = joined[f"{ci_high}_std"].to_numpy().astype(float)
    ci_low_firth = joined[f"{ci_low}_firth"].to_numpy().astype(float)
    ci_high_firth = joined[f"{ci_high}_firth"].to_numpy().astype(float)

    if ci_hr:
        # log-transform HR-scale CIs for comparison
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ci_low_std = np.log(ci_low_std)
            ci_high_std = np.log(ci_high_std)
            ci_low_firth = np.log(ci_low_firth)
            ci_high_firth = np.log(ci_high_firth)

    valid_ci = (
        np.isfinite(ci_low_std) & np.isfinite(ci_high_std)
        & np.isfinite(ci_low_firth) & np.isfinite(ci_high_firth)
    )
    cl_s, ch_s = ci_low_std[valid_ci], ci_high_std[valid_ci]
    cl_f, ch_f = ci_low_firth[valid_ci], ci_high_firth[valid_ci]

    overlap_low = np.maximum(cl_s, cl_f)
    overlap_high = np.minimum(ch_s, ch_f)
    overlap_width = np.maximum(overlap_high - overlap_low, 0.0)
    union_width = np.maximum(ch_s, ch_f) - np.minimum(cl_s, cl_f)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        overlap_frac = np.where(union_width > 0, overlap_width / union_width, np.nan)
    overlap_frac_valid = overlap_frac[np.isfinite(overlap_frac)]

    results["ci_n"] = int(valid_ci.sum())
    results["ci_mean_overlap"] = float(overlap_frac_valid.mean()) if len(overlap_frac_valid) > 0 else np.nan

    # point-in-CI
    e_std_ci = e_std[valid_ci] if not ci_hr else joined[f"{effect}_std"].to_numpy().astype(float)[valid_ci]
    e_firth_ci = e_firth[valid_ci] if not ci_hr else joined[f"{effect}_firth"].to_numpy().astype(float)[valid_ci]
    std_in_firth_ci = ((e_std_ci >= cl_f) & (e_std_ci <= ch_f)).mean()
    firth_in_std_ci = ((e_firth_ci >= cl_s) & (e_firth_ci <= ch_s)).mean()
    results["ci_std_in_firth"] = float(std_in_firth_ci)
    results["ci_firth_in_std"] = float(firth_in_std_ci)

    # --- Bland-Altman ---
    ba_mean = (e_std_v + e_firth_v) / 2
    ba_diff = e_std_v - e_firth_v
    ba_bias = float(ba_diff.mean()) if len(ba_diff) > 0 else np.nan
    ba_sd = float(ba_diff.std(ddof=1)) if len(ba_diff) > 1 else np.nan
    ba_upper = ba_bias + 1.96 * ba_sd if np.isfinite(ba_sd) else np.nan
    ba_lower = ba_bias - 1.96 * ba_sd if np.isfinite(ba_sd) else np.nan
    outside = int(((ba_diff > ba_upper) | (ba_diff < ba_lower)).sum()) if np.isfinite(ba_upper) else 0
    ba_t, ba_t_p = _paired_t_test(ba_diff)

    results["ba_bias"] = ba_bias
    results["ba_sd"] = ba_sd
    results["ba_upper"] = ba_upper
    results["ba_lower"] = ba_lower
    results["ba_outside"] = outside
    results["ba_t"] = ba_t
    results["ba_t_p"] = ba_t_p
    results["_ba_mean"] = ba_mean
    results["_ba_diff"] = ba_diff

    return results


# ---------------------------------------------------------------------------
# Text summary
# ---------------------------------------------------------------------------


def print_summary(pair: dict, joined: pl.DataFrame, std_df: pl.DataFrame,
                  firth_df: pl.DataFrame, m: dict) -> None:
    """Print formatted comparison summary."""
    w = 60
    print()
    print("=" * w)
    print(f"  {pair['label']}")
    print("=" * w)

    n_std_only = len(std_df) - m["n_matched"]
    n_firth_only = len(firth_df) - m["n_matched"]
    print(f"\nData Overview")
    print(f"  Matched phecodes:      {m['n_matched']}")
    print(f"  Only in standard:      {n_std_only}")
    print(f"  Only in firth:         {n_firth_only}")

    print(f"\nConvergence")
    print(f"  Standard converged:    {m['conv_std_count']}/{m['n_matched']} ({m['conv_std_rate']:.1%})")
    print(f"  Firth converged:       {m['conv_firth_count']}/{m['n_matched']} ({m['conv_firth_rate']:.1%})")
    print(f"  Std only converged:    {m['conv_std_only']}")
    print(f"  Firth only converged:  {m['conv_firth_only']}")

    print(f"\nEffect Agreement (n={m['effect_n']})")
    print(f"  Pearson r:             {m['effect_pearson']:.6f}")
    print(f"  Spearman rho:          {m['effect_spearman']:.6f}")
    print(f"  Mean abs difference:   {m['effect_mad']:.6f}")
    print(f"  RMSD:                  {m['effect_rmsd']:.6f}")
    print(f"  Max |diff|:            {m['effect_max_diff']:.6f}  (phecode {m['effect_max_diff_phecode']})")

    print(f"\nP-value Agreement (n={m['pval_n']})")
    print(f"  -log10(p) Pearson r:   {m['pval_pearson']:.6f}")
    print(f"  -log10(p) Spearman:    {m['pval_spearman']:.6f}")
    bonf = m["bonferroni_thresh"]
    print(f"  Bonferroni threshold:  {bonf:.2e}")
    for name in ["0.05", "0.01", "bonferroni"]:
        c = m["concordance"][name]
        tbl = c["table"]
        print(f"\n  Concordance at p < {name}:")
        print(f"                         Firth+    Firth-")
        print(f"    Std+                 {tbl[0,0]:>6d}    {tbl[0,1]:>6d}")
        print(f"    Std-                 {tbl[1,0]:>6d}    {tbl[1,1]:>6d}")
        print(f"    Kappa:               {c['kappa']:.4f}")
        print(f"    McNemar chi2:        {c['mcnemar_chi2']:.4f}  (p={c['mcnemar_p']:.4f})")

    print(f"\nSE Comparison (n={m['se_n']}, excluded SE=0: {m['se_zero_count']})")
    print(f"  Mean ratio (F/S):      {m['se_mean_ratio']:.4f}")
    print(f"  Median ratio (F/S):    {m['se_median_ratio']:.4f}")
    print(f"  Firth SE > Std SE:     {m['se_firth_larger_frac']:.1%}")
    print(f"  Paired t-stat:         {m['se_paired_t']:.4f}  (p={m['se_paired_t_p']:.4f})")

    print(f"\nCI Overlap (n={m['ci_n']})")
    print(f"  Mean overlap fraction: {m['ci_mean_overlap']:.4f}")
    print(f"  Std point in Firth CI: {m['ci_std_in_firth']:.1%}")
    print(f"  Firth point in Std CI: {m['ci_firth_in_std']:.1%}")

    print(f"\nBland-Altman (effect: std - firth)")
    print(f"  Bias (mean diff):      {m['ba_bias']:.6f}")
    print(f"  SD of differences:     {m['ba_sd']:.6f}")
    print(f"  Limits of agreement:   [{m['ba_lower']:.6f}, {m['ba_upper']:.6f}]")
    print(f"  Outside limits:        {m['ba_outside']}/{m['effect_n']}")
    print(f"  Bias t-test:           t={m['ba_t']:.4f}  (p={m['ba_t_p']:.4f})")
    print()


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _save(fig: plt.Figure, path: str) -> None:
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def plot_effect_scatter(m: dict, tag: str, effect_col: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(m["_e_std"], m["_e_firth"], s=8, alpha=0.5, edgecolors="none")
    lims = [
        min(m["_e_std"].min(), m["_e_firth"].min()),
        max(m["_e_std"].max(), m["_e_firth"].max()),
    ]
    ax.plot(lims, lims, "k--", linewidth=0.8, label="identity")
    ax.set_xlabel(f"{effect_col} (standard)")
    ax.set_ylabel(f"{effect_col} (firth)")
    ax.set_title(f"Effect size: {tag}")
    ax.annotate(f"r = {m['effect_pearson']:.4f}", xy=(0.05, 0.93),
                xycoords="axes fraction", fontsize=10)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig


def plot_pvalue_scatter(m: dict, tag: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 6))
    nlp_s, nlp_f = m["_nlp_std"], m["_nlp_firth"]
    bonf_line = -np.log10(m["bonferroni_thresh"])

    # discordant: significant in one but not other at Bonferroni
    disc = (nlp_s >= bonf_line) != (nlp_f >= bonf_line)

    ax.scatter(nlp_s[~disc], nlp_f[~disc], s=8, alpha=0.4, edgecolors="none",
               color="steelblue", label="concordant")
    if disc.any():
        ax.scatter(nlp_s[disc], nlp_f[disc], s=14, alpha=0.7, edgecolors="none",
                   color="crimson", label="discordant")

    lims = [0, max(nlp_s.max(), nlp_f.max()) * 1.05]
    ax.plot(lims, lims, "k--", linewidth=0.8)
    ax.axhline(bonf_line, color="grey", linestyle=":", linewidth=0.7)
    ax.axvline(bonf_line, color="grey", linestyle=":", linewidth=0.7)
    ax.set_xlabel("-log10(p) standard")
    ax.set_ylabel("-log10(p) firth")
    ax.set_title(f"P-value agreement: {tag}")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig


def plot_bland_altman(m: dict, tag: str, effect_col: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(m["_ba_mean"], m["_ba_diff"], s=8, alpha=0.4, edgecolors="none")
    ax.axhline(m["ba_bias"], color="blue", linewidth=0.8, label=f"bias = {m['ba_bias']:.4f}")
    ax.axhline(m["ba_upper"], color="red", linestyle="--", linewidth=0.7,
               label=f"+1.96 SD = {m['ba_upper']:.4f}")
    ax.axhline(m["ba_lower"], color="red", linestyle="--", linewidth=0.7,
               label=f"-1.96 SD = {m['ba_lower']:.4f}")
    ax.set_xlabel(f"Mean {effect_col}")
    ax.set_ylabel(f"Difference (std - firth)")
    ax.set_title(f"Bland-Altman: {tag}")
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    return fig


def plot_se_ratio(m: dict, tag: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 4))
    se_r = m["_se_ratio"]
    bins = min(50, max(10, len(se_r) // 5))
    ax.hist(se_r, bins=bins, edgecolor="white", linewidth=0.3, color="steelblue")
    ax.axvline(1.0, color="red", linestyle="--", linewidth=0.8, label="ratio = 1.0")
    ax.axvline(m["se_mean_ratio"], color="orange", linestyle="-", linewidth=0.8,
               label=f"mean = {m['se_mean_ratio']:.3f}")
    ax.set_xlabel("SE ratio (Firth / Standard)")
    ax.set_ylabel("Count")
    ax.set_title(f"SE ratio distribution: {tag}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_significance_heatmap(m: dict, tag: str) -> plt.Figure:
    c = m["concordance"]["bonferroni"]
    tbl = c["table"]
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(tbl, cmap="Blues", aspect="auto")
    labels = ["Significant", "Not significant"]
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Firth")
    ax.set_ylabel("Standard")
    ax.set_title(f"Significance concordance (Bonferroni): {tag}")
    # annotate cells
    for i in range(2):
        for j in range(2):
            color = "white" if tbl[i, j] > tbl.max() * 0.5 else "black"
            ax.text(j, i, str(tbl[i, j]), ha="center", va="center",
                    fontsize=14, fontweight="bold", color=color)
    ax.annotate(f"kappa = {c['kappa']:.3f}", xy=(0.02, -0.18),
                xycoords="axes fraction", fontsize=9)
    fig.tight_layout()
    return fig


def plot_dashboard(m: dict, tag: str, effect_col: str) -> plt.Figure:
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(f"Comparison dashboard: {tag}", fontsize=14, fontweight="bold")

    # 1. Effect scatter
    ax = axes[0, 0]
    ax.scatter(m["_e_std"], m["_e_firth"], s=6, alpha=0.4, edgecolors="none")
    lims = [min(m["_e_std"].min(), m["_e_firth"].min()),
            max(m["_e_std"].max(), m["_e_firth"].max())]
    ax.plot(lims, lims, "k--", linewidth=0.7)
    ax.set_xlabel(f"{effect_col} (std)")
    ax.set_ylabel(f"{effect_col} (firth)")
    ax.set_title("Effect size")
    ax.annotate(f"r = {m['effect_pearson']:.4f}", xy=(0.05, 0.9), xycoords="axes fraction", fontsize=8)

    # 2. P-value scatter
    ax = axes[0, 1]
    nlp_s, nlp_f = m["_nlp_std"], m["_nlp_firth"]
    bonf_line = -np.log10(m["bonferroni_thresh"])
    disc = (nlp_s >= bonf_line) != (nlp_f >= bonf_line)
    ax.scatter(nlp_s[~disc], nlp_f[~disc], s=6, alpha=0.3, edgecolors="none", color="steelblue")
    if disc.any():
        ax.scatter(nlp_s[disc], nlp_f[disc], s=10, alpha=0.6, edgecolors="none", color="crimson")
    p_lims = [0, max(nlp_s.max(), nlp_f.max()) * 1.05]
    ax.plot(p_lims, p_lims, "k--", linewidth=0.7)
    ax.axhline(bonf_line, color="grey", linestyle=":", linewidth=0.6)
    ax.axvline(bonf_line, color="grey", linestyle=":", linewidth=0.6)
    ax.set_xlabel("-log10(p) std")
    ax.set_ylabel("-log10(p) firth")
    ax.set_title("P-value agreement")

    # 3. Bland-Altman
    ax = axes[0, 2]
    ax.scatter(m["_ba_mean"], m["_ba_diff"], s=6, alpha=0.3, edgecolors="none")
    ax.axhline(m["ba_bias"], color="blue", linewidth=0.7)
    ax.axhline(m["ba_upper"], color="red", linestyle="--", linewidth=0.6)
    ax.axhline(m["ba_lower"], color="red", linestyle="--", linewidth=0.6)
    ax.set_xlabel(f"Mean {effect_col}")
    ax.set_ylabel("Diff (std - firth)")
    ax.set_title("Bland-Altman")

    # 4. SE ratio
    ax = axes[1, 0]
    se_r = m["_se_ratio"]
    bins = min(50, max(10, len(se_r) // 5))
    ax.hist(se_r, bins=bins, edgecolor="white", linewidth=0.3, color="steelblue")
    ax.axvline(1.0, color="red", linestyle="--", linewidth=0.7)
    ax.axvline(m["se_mean_ratio"], color="orange", linewidth=0.7)
    ax.set_xlabel("SE ratio (F/S)")
    ax.set_ylabel("Count")
    ax.set_title("SE ratio distribution")

    # 5. Significance heatmap
    ax = axes[1, 1]
    c = m["concordance"]["bonferroni"]
    tbl = c["table"]
    ax.imshow(tbl, cmap="Blues", aspect="auto")
    labels = ["Sig", "Not sig"]
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Firth")
    ax.set_ylabel("Standard")
    ax.set_title("Concordance (Bonferroni)")
    for i in range(2):
        for j in range(2):
            color = "white" if tbl[i, j] > tbl.max() * 0.5 else "black"
            ax.text(j, i, str(tbl[i, j]), ha="center", va="center",
                    fontsize=12, fontweight="bold", color=color)

    # 6. Summary text
    ax = axes[1, 2]
    ax.axis("off")
    summary_lines = [
        f"Matched phecodes: {m['n_matched']}",
        f"Effect Pearson r: {m['effect_pearson']:.4f}",
        f"Effect RMSD: {m['effect_rmsd']:.4f}",
        f"-log10(p) Pearson r: {m['pval_pearson']:.4f}",
        f"SE mean ratio (F/S): {m['se_mean_ratio']:.4f}",
        f"CI mean overlap: {m['ci_mean_overlap']:.4f}",
        f"Bland-Altman bias: {m['ba_bias']:.4f}",
        f"Conv. std: {m['conv_std_rate']:.1%}  firth: {m['conv_firth_rate']:.1%}",
    ]
    ax.text(0.05, 0.95, "\n".join(summary_lines), transform=ax.transAxes,
            fontsize=10, verticalalignment="top", family="monospace",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
    ax.set_title("Summary")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def generate_plots(m: dict, pair: dict) -> None:
    """Generate and save all 6 plots for a pair."""
    tag = pair["tag"]
    effect_col = pair["effect_col"]

    plots = {
        f"{tag}_effect_scatter.png": plot_effect_scatter(m, tag, effect_col),
        f"{tag}_pvalue_scatter.png": plot_pvalue_scatter(m, tag),
        f"{tag}_bland_altman.png": plot_bland_altman(m, tag, effect_col),
        f"{tag}_se_ratio.png": plot_se_ratio(m, tag),
        f"{tag}_significance_heatmap.png": plot_significance_heatmap(m, tag),
        f"{tag}_dashboard.png": plot_dashboard(m, tag, effect_col),
    }

    for filename, fig in plots.items():
        path = os.path.join(PLOT_DIR, filename)
        _save(fig, path)
        print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    os.makedirs(PLOT_DIR, exist_ok=True)

    any_processed = False
    for pair in PAIRS:
        loaded = load_pair(pair)
        if loaded is None:
            continue

        std_df, firth_df, joined = loaded
        if len(joined) == 0:
            print(f"WARNING: No matched phecodes for {pair['label']} — skipping.")
            continue

        any_processed = True
        m = compute_metrics(pair, joined)
        print_summary(pair, joined, std_df, firth_df, m)
        generate_plots(m, pair)

    if not any_processed:
        print("No pairs could be compared. Run benchmark/run_benchmark.py first.")
        sys.exit(1)

    print(f"\nAll plots saved to {PLOT_DIR}/")


if __name__ == "__main__":
    main()
