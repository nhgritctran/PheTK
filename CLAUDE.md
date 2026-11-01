# PheTK - Phenotype Toolkit

Fast Python library for Phenome-Wide Association Studies (PheWAS), supporting phecode 1.2 and phecodeX 1.0. Designed primarily for the All of Us Research Program but supports custom datasets.

- **Version**: 0.3.7
- **Author**: Tam C Tran (PheTK@mail.nih.gov)
- **Repo**: https://github.com/nhgritctran/PheTK
- **License**: GPL-3.0
- **Published**: Bioinformatics, Jan 2025 — doi:10.1093/bioinformatics/btae719

## Package Structure

```
src/phetk/
├── __main__.py     # CLI entry point (argparse subcommands)
├── clinvar.py      # ClinVar class: variant retrieval by gene/region from the ClinVar VCF
├── cohort.py       # Cohort class: genotype-based cohort & covariates
├── phewas.py       # PheWAS class: logistic & Cox regression analysis
├── phecode.py      # Phecode class: ICD-to-phecode mapping & time-to-event
├── plot/           # Plot package: pluggable backend architecture
│   ├── __init__.py   # Plot facade class + backend imports
│   ├── _base.py      # PlotBackend ABC, PlotContext dataclass, @register, get_plot_backend()
│   ├── _shared.py    # Shared helper functions (stateless utilities)
│   ├── _manhattan.py  # @register("manhattan") ManhattanBackend
│   ├── _miami.py      # @register("miami") MiamiBackend
│   ├── _volcano.py    # @register("volcano") VolcanoBackend
│   └── _forest.py     # @register("forest") ForestBackend
├── demo.py         # Quick demo with synthetic data
├── _utils.py       # Dataframe conversion, BigQuery helpers, phecode loading
├── _queries.py     # BigQuery SQL templates for All of Us OMOP
├── _paths.py       # GCS paths for AoU ancestry & genotype data
├── _dsub.py        # Google Cloud dsub wrapper for distributed jobs
└── phecode/        # Phecode mapping CSV files (phecode12, phecodeX, phecodeX_WHO)
```

## Key Dependencies

- **Python**: >=3.8 (Docker image uses 3.10)
- **lifelines**: >=0.28.0,<0.30.1 (pinned for Python 3.10 compatibility — Cox regression)
- **numpy**: >=1.21.0,<2.0
- **polars**: fast dataframes (preferred over pandas internally)
- **hail**: genomic data processing (All of Us)
- **google-cloud-bigquery**, **google-cloud-storage**: All of Us OMOP access
- **statsmodels**: logistic regression
- **matplotlib**, **adjusttext**: visualization
- **tqdm**: progress bars
- **pysam**: remote tabix range reads of the ClinVar VCF (`clinvar.py`)

## CLI Commands

```bash
phetk demo
phetk phewas [args]
phetk cohort by-genotype [args]
phetk cohort add-covariates [args]
phetk phecode count-phecode [args]
phetk phecode add-age-at-first-event [args]
phetk phecode add-phecode-time-to-event [args]
phetk clinvar search [args]
```

## Install & Run

```bash
pip install phetk              # from PyPI
pip install -e .               # local development install
python -m phetk demo           # quick smoke test
```

Docker: `docker pull phetk/phetk:latest`

## Analysis Workflow

0. **ClinVar** (`clinvar.py`, optional): Get a variant list for a gene/region to feed into `Cohort.by_genotype()`
1. **Cohort** (`cohort.py`): Build cohort by genotype variant, add covariates (age, sex, ancestry PCs, EHR metrics)
2. **Phecode** (`phecode.py`): Extract ICD codes from OMOP, map to phecodes, compute counts / age-at-first-event / time-to-event
3. **PheWAS** (`phewas.py`): Run logistic or Cox regression across all phecodes (parallel via ThreadPoolExecutor / ProcessPoolExecutor)
4. **Plot** (`plot/`): Manhattan (effect direction markers), Miami (mirrored Manhattan, markers sized by effect magnitude), or forest (CIs)

## Phecode Versions

| Version | `phecode_version` string | `icd_version` | File | Notes |
|---|---|---|---|---|
| Phecode 1.2 | `"1.2"` | `"US"` | `phecode12.csv` | Supports exclusion ranges |
| PhecodeX 1.0 | `"X1.0"` | `"US"` | `phecodeX.csv` | US ICD mapping |
| PhecodeX 1.0 | `"X1.0"` | `"WHO"` | `phecodeX_WHO.csv` | WHO ICD mapping |
| — | `"X"` | — | — | Alias for the latest phecodeX release |

`phecode_version` is case/whitespace-insensitive (`"x1.0"`, `" X "` both work); `icd_version` is matched strictly.

### Registry (`_utils.py`)

`_utils.py` owns the mapping registry — `LATEST_PHECODE_X_VERSION`, `PHECODE_MAP_FILES`,
`PHECODE_MAP_SCHEMAS`, `PHECODE_VERSION_ALIASES` — plus the resolvers
(`resolve_phecode_version`, `phecode_version_family`, `available_phecode_versions`,
`accepted_phecode_versions`, `available_icd_versions`). It must live there because
`phecode.py` imports `_utils`, so the reverse would be a circular import.

Loader layering:

```
_utils.load_phecode_map()          raises ValueError   <- core, all logic here
├── _utils.get_phecode_mapping_table()   sys.exit(1)   <- CLI wrapper
│     └── used by count_phecode(), PheWAS.__init__()
└── phecode.get_phecode_map()      raises ValueError   <- public API
```

Internal branch points use `_utils.phecode_version_family()` (`"1.2"` or `"X"`), never the
raw string, so pinned releases take the same code paths as their family.

**Adding a new phecodeX release (e.g. X1.1):**
1. Drop the CSV into `src/phetk/phecode/` (already packaged by the `phetk = ["phecode/*"]` glob in `pyproject.toml`).
2. Add the `(version, icd_version) -> filename` entries to `PHECODE_MAP_FILES`.
3. Add a `PHECODE_MAP_SCHEMAS` entry (copy X1.0's if the schema is unchanged).
4. Bump `LATEST_PHECODE_X_VERSION` so the `"X"` alias points at the new release.

No other PheTK code needs to change.

## ClinVar Module

`clinvar.py` reads NCBI's ClinVar VCF via remote tabix (`pysam`) rather than E-utilities:
`esummary` returns rsIDs for only ~3% of records, never returns allele frequencies, and
leaves ref/alt empty on most records. Entrez also silently rewrites unknown field tags (e.g.
`[Germline classification]`) into free-text searches, so client-side filtering on VCF INFO
fields is both more correct and keeps the vocabulary under our control.

Three parsing traps, all covered by `tests/unit/test_clinvar_unit.py`:

1. **Compound `CLNSIG`.** Values like `Pathogenic/Likely_pathogenic` and
   `Pathogenic|drug_response` are split on `|` and `/` into a component set, so
   `clinical_significance="Pathogenic"` matches them. `Conflicting classifications of
   pathogenicity` has no separator and stays intact.
2. **pysam splits `Number=.` INFO fields on commas.** `criteria_provided,_single_submitter`
   arrives as `('criteria_provided', '_single_submitter')`. `_norm_info()` rejoins on `,` then
   maps `_` → space.
3. **`CLNSIG` is absent on ~3.5% of records.** The normalizer returns `None`, never raises.

Review stars are derived from the review status string (`REVIEW_STATUS_STARS`), not stored.
`min_review_star` defaults to `None`, so 0-star records are kept.

**Remote index handling (`_fetch_remote_index`).** Both ClinVar releases are named
`clinvar.vcf.gz`. htslib caches a remote `.tbi` into the CWD under that basename and reuses it
without validating it against the VCF, so GRCh38 and GRCh37 collide (second one opened fails
with `Invalid BGZF header`), as do successive weekly releases. PheTK downloads the index
itself into a per-process temp dir under a URL-derived name. Do not "simplify" this back to a
bare `pysam.VariantFile(url)` — `TestRemoteIndexHandling` guards it.

Allele frequencies (`AF_ESP`/`AF_EXAC`/`AF_TGP`) are **global**; ClinVar has no
ancestry-stratified frequencies. Use gnomAD for that — out of scope here.

**Hand-off contract.** Four output columns are named after `Cohort.by_genotype()`'s
parameters — `chromosome_number`, `genomic_position`, `ref_allele`, `alt_allele` — so a row
splats straight in. `TestByGenotypeHandoff` asserts this against
`inspect.signature(Cohort.by_genotype)`; renaming either side fails the suite.

`chromosome_number` is a **string** column, unprefixed, so `X`/`Y` survive. Do not add a `chr`
prefix: `by_genotype()` prepends it itself on GRCh38 (`cohort.py:206`, `:442`, `:585`), so
`"chr7"` would yield `chrchr7`. Console output uses `chr7:...` on GRCh38 via `_display_locus()`,
but that is display only. The `--chromosome_number` CLI flag (`type=int`) still can't take X/Y.

Error style follows the `load_phecode_map` / `get_phecode_mapping_table` convention: the
public API raises `ValueError`, `main_search()` catches, prints, and `sys.exit(1)`.

### Test markers

`network`-marked tests are skipped unless `PHETK_NETWORK_TESTS` is set (mirroring the `aou` /
`WORKSPACE_CDR` hook in `tests/conftest.py`), so CI stays hermetic.

```bash
python -m pytest tests/ -m "not network"          # hermetic (CI default)
PHETK_NETWORK_TESTS=1 python -m pytest -m network # live NCBI
```

## Platform Compatibility

PheTK is primarily designed for the **All of Us (AoU) Researcher Workbench**, a closed GCP environment. Some tools work outside it; others will silently fail without the right env vars or GCP access.

### AoU-only (require Workbench environment)

| Tool | Why AoU-only |
|---|---|
| `Cohort.by_genotype()` | Needs Hail + hardcoded GCS paths in `_paths.py` (`fc-aou-datasets-controlled`) |
| `Cohort._get_ancestry_preds()` | Reads AoU ancestry TSV files from GCS (CDR v6–8) |
| `Cohort._get_covariates()` / `add_covariates()` | Runs BigQuery OMOP queries via `_queries.py` |
| `Phecode.__init__(platform="aou")` | Reads ICD codes from OMOP via BigQuery at init time |
| `Phecode.add_age_at_first_event()` | BigQuery query; requires `WORKSPACE_CDR` env var |
| `PheWAS.run_dsub()` | Needs dsub + Google Batch + `WORKSPACE_BUCKET` |
| All of `_queries.py` | OMOP schema queries (condition_occurrence, person, observation, concept tables) |
| All of `_dsub.py` | Google Batch/dsub + AoU-specific env vars |
| `_paths.py` | All paths hardcoded to `gs://fc-aou-datasets-controlled/` |
| `_utils.polars_gbq()`, `spark_to_polars()`, `polars_gbq_chunk()` | Require BigQuery / Spark runtime |

### Cross-platform (work outside AoU)

| Tool | Notes |
|---|---|
| `phecode.get_phecode_map()` | Loads a bundled/custom mapping table from local CSVs; raises `ValueError` |
| `Phecode.count_phecode()` | Maps ICD→phecodes using local CSV mapping files |
| `Phecode.add_phecode_time_to_event()` | Fully static/file-based |
| `PheWAS.run()` | Local logistic/Cox regression on pre-loaded data |
| `Plot` (all methods) | Local plotting only |
| `demo.run()` | Synthetic data, no cloud needed |
| `ClinVar.search()` | Needs outbound HTTPS to `ftp.ncbi.nlm.nih.gov` (and `eutils.ncbi.nlm.nih.gov` when `gene=` is used). On AoU, egress may be blocked: stage `clinvar.vcf.gz` + `.tbi` into the workspace bucket and pass `vcf_path`, then use `region=` instead of `gene=`. Raises `ValueError` |
| `clinvar.get_gene_region()` | NCBI Gene lookup; in-process cached, throttled, honors `NCBI_API_KEY` |

### Key platform parameters

```
platform="aou"    → activates BigQuery/OMOP/Hail paths (default)
platform="custom" → requires gbq_dataset_id (user's BigQuery dataset with OMOP schema)
icd_file_path     → local/custom ICD file bypass (skips OMOP query entirely)
```

### AoU environment variables

These must exist inside the Workbench. Code will break silently if run outside without them:

| Variable | Used by |
|---|---|
| `WORKSPACE_CDR` | OMOP dataset ID for BigQuery queries |
| `GOOGLE_PROJECT` | GCP billing project |
| `WORKSPACE_BUCKET` | GCS bucket for dsub I/O |
| `OWNER_EMAIL` | dsub job ownership / naming |
| `ARTIFACT_REGISTRY_DOCKER_REPO` | AoU Docker image registry prefix |

- **dsub**: `_dsub.py` submits distributed jobs to Google Cloud; see `docs/dsub-considerations.md`

### Verily Workbench env setup (`_utils.setup_verily_env`)

On Verily, the variables above are derived from the `wb` CLI instead of being pre-set.
`_classify_wb_buckets` splits `GCS_BUCKET` resources by `stewardshipType` — CONTROLLED →
`WORKSPACE_BUCKET`, REFERENCED → `WORKSPACE_REFERENCED_BUCKET` — after dropping AoU data and
workspace infrastructure via `_is_user_bucket`.

**AoU buckets are identified by `bucketName`, not by stewardship and not by resource id.**
Stewardship can't do it: a user's cross-workspace reference and the AoU All by All bucket are
both `REFERENCED`. Resource id can't do it either — it is a label chosen at registration time,
and AoU registered All by All as `vwb_aou_allxall_v8` (underscores) pointing at bucket
`vwb-aou-allxall` (hyphens). A prefix test on the id alone therefore missed it and leaked the
bucket into `WORKSPACE_REFERENCED_BUCKET2`. `_is_user_bucket` checks `bucketName` first, falls
back to the resource id, and matches `^vwb[-_]aou[-_]` so either separator is caught.
`_classify_gcloud_buckets` uses the same predicate so the two discovery paths can't drift.
Covered by `tests/unit/test_verily_env.py`.

**Numbered variables use an underscore**: `WORKSPACE_BUCKET_1..N` and
`WORKSPACE_REFERENCED_BUCKET_1..N`, set only when more than one bucket survives filtering
(a single survivor gets the unnumbered name). `_print_env_summary` discovers them by probing
`f"{var}_{i}"` and stopping at the first gap, so renaming the setters without the probe
silently drops them from the printed summary.

Caveat: with multiple controlled buckets, plain `WORKSPACE_BUCKET` is deliberately left unset
so the user picks one — but `_dsub.py` reads `WORKSPACE_BUCKET`, so `run_dsub()` will fail
until they do.

## Writing output to GCS (`_utils.write_tsv`)

### Destination classification (`_to_gs_uri_if_bucket_mount`)

`write_tsv` expands `~` and resolves the path to absolute form first, so the destination
depends only on where a path points, never on how it was spelled — a bare filename, a `../`
path and an absolute path to the same place all classify identically. Then:

| Kind | Recognized by | Attempts |
|---|---|---|
| `gs://` URI | caller passed one | 1-3 below |
| gcsfuse mount path | a **verified** mount | 1-3, then POSIX write through the mount |
| local | everything else | atomic staged write (`_write_local`) |

A mount is verified by two independent signals, both requiring a real mount: the kernel mount
table (`_gcsfuse_mounts()` parses `/proc/mounts`, which names the bucket directly and works
wherever it is mounted), then Verily's `/home/jupyter/workspace/<bucket-name>` convention
confirmed with `os.path.ismount` — a fallback for when the mount is invisible in this
process's namespace. Nested mounts resolve longest-mountpoint-first; `--only-dir` mounts carry
their prefix into the key.

Do not reintroduce a bare string-prefix test here. The original one treated the first segment
under `/home/jupyter/workspace/` as a bucket name with no mount check, so a persistent-disk
directory was redirected to a real bucket the caller had no IAM on, and all write methods
403'd on a write that needed no network. It also silently excluded AoU, whose path is
`workspaces` (**plural**) — hence "relative paths always worked on AoU but not Verily".
`TestToGsUriIfBucketMount` guards every case.

### Fallback tiers

Three methods are tried in order — direct Polars write, `gcsfs` streaming, local staging +
`gcloud storage cp` — because a workspace service account often lacks
`storage.multipartUploads.create` (needed by the first) while still being able to stream.

For a mount path a fourth is appended: the POSIX write through the mount itself. All three GCS
tiers authenticate as the end-user account, which on a migrated workspace may hold no
`storage.objects.create` on the bucket, while the mount was made with a credential that can
write. It stays **last** because gcsfuse writes are unreliable for medium-to-large files (EIO
through its filesystem layer — the reason commit `0a9fe22` started rewriting mount paths to
`gs://` at all), and because a permission error on the GCS routes fails in milliseconds while
a FUSE write fails slowly. An explicit `gs://` URI has no filesystem equivalent, so it never
gets this tier.

Falling back is routine, so the reporting is deliberately quiet:

- First method works → **no output at all**.
- A method fails → one line naming **only the exception type** and the next method to be
  tried, then `Saved using <method>.` on success.
- **All** methods fail → `RuntimeError` listing every method with its **full** message,
  chained via `raise ... from` to the last exception. A rewritten mount path also names the
  derived destination (`Could not write to <path> (resolved to gs://...)`), so a redirect is
  never a mystery.

Exception messages are withheld from the fallback notice on purpose. Cloud storage errors
embed XML, URL-encoded paths, bucket names and service account emails; printing them between
"Writing output to ..." and "Successfully generated ..." made a successful write look like a
hard failure. Do not "improve" this by truncating the message instead — truncation still
leaks paths and identifiers, and `TestWriteTsvOutput.test_no_part_of_the_message_is_printed`
scans for every token of the message in the output.

`TestWriteTsvFallback` covers which method runs; `TestWriteTsvOutput` covers what is printed.
Keep the two concerns separate — the write calls themselves are known-good and should not be
altered for cosmetic reasons.

Related: `count_phecode()` issues `PRAGMA disable_progress_bar` on its DuckDB connection.
DuckDB renders progress as an ipywidget, and on AoU/Verily the widget model is frequently
missing, so notebooks showed "Error displaying widget: model not found" instead of progress.
PheTK prints its own progress lines.

## Docs & Tutorials

- `docs/clinvar-module.md`, `cohort-module.md`, `phecode-module.md`, `phewas-module.md`, `plot-module.md`
- `docs/tutorials/` — 4 Jupyter notebooks (logistic, Cox, CLI, dsub) for All of Us Workbench
- `docs/tutorials/README_FIRST.md` — start here for tutorials

## Resource Recommendations

| Workload | Machine |
|---|---|
| Logistic regression (small) | c2d-highcpu-4 (4 vCPU, 8 GB) |
| Cox regression | c2d-standard-4 (4 vCPU, 16 GB) |
| `count_phecode()`, AoU v8 full cohort, `engine="duckdb"` (default) | 4 vCPU, 26 GB RAM |
| `count_phecode()`, AoU v8 full cohort, `engine="polars"` | 16 vCPU, 104 GB RAM |

## Code Style

- **Docstrings**: Use [Google style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings). Do not use reST (`:param`/`:type`/`:return`/`:rtype:`). Types are already in annotations, so omit them from docstrings. Example:

```python
def example(name: str, count: int = 1) -> bool:
    """
    Summary line.

    Extended description if needed.

    Args:
        name: Description of name.
        count: Description of count.

    Returns:
        Description of return value.
    """
```

## Dev Container

`.devcontainer/devcontainer.json` targets All of Us Researcher Workbench (image: `phetk/phetk:latest`). Not for general local dev.