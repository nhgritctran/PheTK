"""
Generate a realistic mock PheWAS results file for plot testing.

Produces a TSV that mimics logistic regression output with ~2500 phecodes
spread across all 18 phecodeX categories. The distribution is designed to
exercise every plot feature:

    Distribution (~2500 phecodes):
    ├── ~80%  non-significant   neg_log_p ∈ [0, nominal)
    ├── ~10%  nominally sig.    neg_log_p ∈ [nominal, bonferroni)
    ├── ~8%   Bonferroni sig.   neg_log_p ∈ [bonferroni, 20]
    ├── 3     infinite p-value  neg_log_p = inf
    └── ~2%   non-converged

    OR spectrum (symmetric magnitude via max(OR, 1/OR)):
    ├── majority near 1         OR ∈ [0.8, 1.25]
    ├── modest                  OR ∈ [0.5, 0.8) ∪ (1.25, 2]
    ├── moderate                OR ∈ [0.2, 0.5) ∪ (2, 5]
    ├── large                   OR ∈ [0.1, 0.2) ∪ (5, 10]
    └── very large (>cap)       OR ∈ (0, 0.1) ∪ (10, 50]

    Edge cases covered:
    - Infinity p-values (3 phecodes)
    - Non-converged results (~2%)
    - Sex restrictions (Both, Male, Female)
    - All 18 phecodeX categories represented
    - Points just above and just below Bonferroni
    - Points just above and just below nominal significance
    - Very small case counts (cases ≤ 10)
    - Very large case counts (cases > 1000)
    - beta exactly 0 (OR = 1)
    - Extremely large OR (>10, tests magnitude cap)
    - Extremely small OR (<0.1, tests magnitude cap on negative side)

Usage:
    python tests/fixtures/generate_mock_phewas_results.py
"""
import os
import sys

import numpy as np
import polars as pl

# Ensure the package is importable when run from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))


def generate() -> pl.DataFrame:
    rng = np.random.default_rng(seed=42)

    # Load real phecodeX mapping for authentic categories / strings / sex
    mapping_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "src", "phetk", "phecode", "phecodeX.csv"
    )
    mapping = pl.read_csv(
        mapping_path,
        schema_overrides={"phecode": str},
        infer_schema_length=10000,
    )
    # Deduplicate to one row per phecode (many ICD codes map to same phecode)
    phecode_info = (
        mapping.select("phecode", "phecode_string", "phecode_category", "sex")
        .unique(subset=["phecode"])
        .sort("phecode")
    )

    # Sample ~2500 phecodes, ensuring every category is represented
    target_n = 2500
    categories = phecode_info["phecode_category"].unique().sort().to_list()

    # Guarantee ≥ 30 per category, fill remainder randomly
    sampled_frames = []
    min_per_cat = 30
    for cat in categories:
        cat_df = phecode_info.filter(pl.col("phecode_category") == cat)
        n_sample = min(min_per_cat, len(cat_df))
        idx = rng.choice(len(cat_df), size=n_sample, replace=False)
        sampled_frames.append(cat_df[idx.tolist()])

    base = pl.concat(sampled_frames)
    remaining_pool = phecode_info.filter(~pl.col("phecode").is_in(base["phecode"]))
    n_remaining = target_n - len(base)
    idx = rng.choice(len(remaining_pool), size=n_remaining, replace=False)
    sampled = pl.concat([base, remaining_pool[idx.tolist()]]).unique(subset=["phecode"]).sort("phecode")
    n = len(sampled)

    # ------------------------------------------------------------------
    # Significance tiers
    # ------------------------------------------------------------------
    bonferroni = -np.log10(0.05 / n)  # ~4.70 for n≈2500

    # Assign each phecode to a significance tier
    tier = rng.choice(
        ["non_sig", "nominal", "bonferroni", "inf"],
        size=n,
        p=[0.80, 0.10, 0.097, 0.003],
    )

    # Generate neg_log_p_value per tier
    nominal = -np.log10(0.05)  # ≈ 1.301
    neg_log_p = np.empty(n, dtype=np.float64)
    for i in range(n):
        if tier[i] == "non_sig":
            neg_log_p[i] = rng.uniform(0, nominal)
        elif tier[i] == "nominal":
            neg_log_p[i] = rng.uniform(nominal, bonferroni)
        elif tier[i] == "bonferroni":
            neg_log_p[i] = rng.uniform(bonferroni, 20)
        else:  # inf
            neg_log_p[i] = np.inf

    # Sprinkle exact boundary values for edge-case testing
    # Just below and just above Bonferroni
    boundary_candidates = np.where(tier == "nominal")[0]
    if len(boundary_candidates) >= 2:
        neg_log_p[boundary_candidates[0]] = bonferroni - 0.01
        neg_log_p[boundary_candidates[1]] = bonferroni + 0.01
    # Just below and just above nominal
    nonsig_candidates = np.where(tier == "non_sig")[0]
    if len(nonsig_candidates) >= 2:
        neg_log_p[nonsig_candidates[0]] = nominal - 0.01
        neg_log_p[nonsig_candidates[1]] = nominal + 0.01

    p_value = np.where(np.isinf(neg_log_p), 0.0, 10.0 ** (-neg_log_p))

    # ------------------------------------------------------------------
    # OR / beta spectrum
    # ------------------------------------------------------------------
    # We want the OR distribution weighted toward 1 but with tails:
    #   60% near 1      [0.8, 1.25]
    #   20% modest       [0.5, 0.8) ∪ (1.25, 2]
    #   10% moderate     [0.2, 0.5) ∪ (2, 5]
    #    6% large        [0.1, 0.2) ∪ (5, 10]
    #    4% very large   (0, 0.1) ∪ (10, 50]
    or_tier = rng.choice(
        ["near1", "modest", "moderate", "large", "very_large"],
        size=n,
        p=[0.60, 0.20, 0.10, 0.06, 0.04],
    )

    odds_ratio = np.empty(n, dtype=np.float64)
    for i in range(n):
        sign = rng.choice([-1, 1])
        if or_tier[i] == "near1":
            mag = rng.uniform(0.8, 1.25)
        elif or_tier[i] == "modest":
            mag = rng.uniform(1.25, 2.0)
        elif or_tier[i] == "moderate":
            mag = rng.uniform(2.0, 5.0)
        elif or_tier[i] == "large":
            mag = rng.uniform(5.0, 10.0)
        else:  # very_large
            mag = rng.uniform(10.0, 50.0)

        odds_ratio[i] = mag if sign == 1 else 1.0 / mag

    # Force a few exact-edge ORs
    # Exactly 1.0 (beta = 0)
    odds_ratio[0] = 1.0
    # Exactly 10 (boundary of default cap)
    odds_ratio[1] = 10.0
    # Exactly 0.1 (1/10, boundary of cap on negative side)
    odds_ratio[2] = 0.1
    # Very extreme values
    odds_ratio[3] = 50.0
    odds_ratio[4] = 0.02  # 1/50

    beta = np.log(odds_ratio)
    log10_or = np.log10(odds_ratio)

    # ------------------------------------------------------------------
    # Standard errors & confidence intervals
    # ------------------------------------------------------------------
    # SE roughly anti-correlated with significance for realism
    se = np.abs(beta) / np.sqrt(np.where(np.isinf(neg_log_p), 30.0, np.maximum(neg_log_p, 0.5)) * 2)
    se = np.clip(se, 0.01, 2.0)
    conf_int_1 = beta - 1.96 * se
    conf_int_2 = beta + 1.96 * se

    # ------------------------------------------------------------------
    # Cases / controls
    # ------------------------------------------------------------------
    cases = np.empty(n, dtype=int)
    case_tier = rng.choice(["tiny", "small", "medium", "large"], size=n, p=[0.05, 0.30, 0.50, 0.15])
    for i in range(n):
        if case_tier[i] == "tiny":
            cases[i] = rng.integers(5, 11)
        elif case_tier[i] == "small":
            cases[i] = rng.integers(11, 100)
        elif case_tier[i] == "medium":
            cases[i] = rng.integers(100, 500)
        else:
            cases[i] = rng.integers(500, 5000)
    controls = rng.integers(2000, 10000, size=n)

    # ------------------------------------------------------------------
    # Convergence
    # ------------------------------------------------------------------
    converged = np.where(rng.random(n) < 0.02, "false", "true")
    # Ensure at least 3 non-converged
    non_conv_idx = rng.choice(n, size=max(3, int(0.02 * n)), replace=False)
    converged[non_conv_idx] = "false"

    # ------------------------------------------------------------------
    # Assemble
    # ------------------------------------------------------------------
    result = sampled.with_columns(
        pl.Series("p_value", p_value),
        pl.Series("neg_log_p_value", neg_log_p),
        pl.Series("beta", beta),
        pl.Series("standard_error", se),
        pl.Series("conf_int_1", conf_int_1),
        pl.Series("conf_int_2", conf_int_2),
        pl.Series("odds_ratio", odds_ratio),
        pl.Series("log10_odds_ratio", log10_or),
        pl.Series("cases", cases),
        pl.Series("controls", controls),
        pl.Series("converged", converged),
    ).rename({"sex": "phecode_sex_restriction"})

    # Reorder columns to match real PheWAS output
    col_order = [
        "phecode", "p_value", "neg_log_p_value", "standard_error",
        "beta", "conf_int_1", "conf_int_2",
        "odds_ratio", "log10_odds_ratio",
        "converged", "cases", "controls",
        "phecode_string", "phecode_category", "phecode_sex_restriction",
    ]
    return result.select(col_order)


def main():
    df = generate()
    out_path = os.path.join(os.path.dirname(__file__), "mock_phewas_results.tsv")
    df.write_csv(out_path, separator="\t")

    # Summary
    bonferroni = -np.log10(0.05 / len(df))
    nominal = -np.log10(0.05)
    n_inf = df.filter(pl.col("neg_log_p_value") == np.inf).height
    n_bonf = df.filter(
        (pl.col("neg_log_p_value") >= bonferroni) & (pl.col("neg_log_p_value") != np.inf)
    ).height
    n_nominal = df.filter(
        (pl.col("neg_log_p_value") >= nominal) & (pl.col("neg_log_p_value") < bonferroni)
    ).height
    n_nonsig = df.filter(pl.col("neg_log_p_value") < nominal).height
    n_nonconv = df.filter(pl.col("converged") == "false").height

    or_vals = df["odds_ratio"]
    magnitude = np.maximum(or_vals.to_numpy(), 1.0 / or_vals.to_numpy())

    print(f"Written to {out_path}")
    print(f"  Total phecodes:     {len(df)}")
    print(f"  Bonferroni:         {bonferroni:.2f}  (-log10(0.05/{len(df)}))")
    print(f"  Non-significant:    {n_nonsig}  ({100*n_nonsig/len(df):.1f}%)")
    print(f"  Nominal only:       {n_nominal}  ({100*n_nominal/len(df):.1f}%)")
    print(f"  Bonferroni sig:     {n_bonf}  ({100*n_bonf/len(df):.1f}%)")
    print(f"  Infinite p-value:   {n_inf}")
    print(f"  Non-converged:      {n_nonconv}")
    print(f"  Categories:         {df['phecode_category'].n_unique()}")
    print(f"  Sex restrictions:   {df['phecode_sex_restriction'].unique().sort().to_list()}")
    print(f"  OR range:           [{or_vals.min():.4f}, {or_vals.max():.2f}]")
    print(f"  Magnitude > 10:     {int(np.sum(magnitude > 10))}")
    print(f"  beta == 0:          {df.filter(pl.col('beta') == 0).height}")


if __name__ == "__main__":
    main()
