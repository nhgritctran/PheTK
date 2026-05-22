"""
Benchmark: generate 10k-person mock cohort and run all 4 regression methods.
Saves PheWAS results and a runtime summary.
"""
import os
import sys
import time
from datetime import datetime

# run from this directory so output files land here
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from phetk._utils import generate_mock_phewas_data
from phetk.phewas import PheWAS

COHORT_SIZE = 10_000
METHODS = ["logit", "cox", "firth_logit", "firth_cox"]
SEED = 42

# --- generate data -----------------------------------------------------------
print(f"Generating mock cohort (n={COHORT_SIZE}, seed={SEED})...")
t0 = time.perf_counter()
generate_mock_phewas_data(cohort_size=COHORT_SIZE, seed=SEED)
data_gen_time = time.perf_counter() - t0
print(f"Data generation: {data_gen_time:.2f}s\n")

# --- run each method ----------------------------------------------------------
timings = {"data_generation": data_gen_time}

for method in METHODS:
    out_file = f"phewas_results_{method}.tsv"
    print(f"Running {method}...")

    phewas_kwargs = {
        "cohort_file_path": "example_cohort.tsv",
        "phecode_count_file_path": "example_phecode_counts.tsv",
        "phecode_version": "X",
        "sex_at_birth_col": "sex",
        "covariate_cols": ["age", "sex", "pc1", "pc2", "pc3"],
        "independent_variable_of_interest": "independent_variable_of_interest",
        "min_cases": 50,
        "min_phecode_count": 2,
        "output_file_path": out_file,
        "method": method,
    }
    if method in ("cox", "firth_cox"):
        phewas_kwargs["cox_control_observed_time_col"] = "observed_time"
        phewas_kwargs["cox_phecode_observed_time_col"] = "phecode_observed_time"

    phewas = PheWAS(**phewas_kwargs)

    t0 = time.perf_counter()
    phewas.run()
    elapsed = time.perf_counter() - t0

    timings[method] = elapsed
    print(f"  {method}: {elapsed:.2f}s  ->  {out_file}\n")

# --- write runtime summary ----------------------------------------------------
summary_path = "benchmark_runtimes.txt"
with open(summary_path, "w") as f:
    f.write(f"PheTK Benchmark — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Cohort size: {COHORT_SIZE}\n")
    f.write(f"Seed: {SEED}\n")
    f.write(f"{'='*50}\n\n")
    f.write(f"{'Step':<25} {'Runtime (s)':>12}\n")
    f.write(f"{'-'*25} {'-'*12}\n")
    for step, secs in timings.items():
        f.write(f"{step:<25} {secs:>12.2f}\n")
    total = sum(timings.values())
    f.write(f"\n{'Total':<25} {total:>12.2f}\n")

print(f"Runtime summary saved to {summary_path}")
