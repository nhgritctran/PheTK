# Phecode Module

Extract ICD codes and map them to phecodes for phenome-wide association studies.

### Recommended VM Configuration
For the full _All of Us_ CDR v8 cohort, a **4 CPU / 26 GB RAM** VM with `engine="duckdb"` has been tested and is sufficient for `count_phecode`. DuckDB keeps memory bounded via its spill-capable pipeline, so this modest config is enough even on the full v8 cohort.
Polars is a bit faster, but requires more RAM, e.g. **16 CPU / 104 GB RAM** for the full v8 cohort.

## get_phecode_map

Load a bundled ICD→phecode mapping table as a polars DataFrame, for inspection, joins, or export.

```python
get_phecode_map(
    phecode_version="X",
    icd_version="US",
    phecode_map_file_path=None,
    keep_all_columns=True,
) -> polars.DataFrame
```

### Key Parameters
- `phecode_version`: Phecode version to load (str, default: `"X"`). Case and surrounding whitespace are ignored, so `"x1.0"` and `" X "` both work.
- `icd_version`: ICD mapping version, `"US"`, `"WHO"`, or `"custom"` (str, default: `"US"`). Matched exactly — `"us"` is rejected.
- `phecode_map_file_path`: Path to a custom mapping file (str, optional). Required when `icd_version="custom"`; otherwise it overrides the bundled file. The file is parsed with the schema of `phecode_version`.
- `keep_all_columns`: If `False`, return only the core mapping columns (bool, default: `True`).

### Supported Versions

| `phecode_version` | `icd_version` | Mapping file |
|---|---|---|
| `"1.2"` | `"US"` | `phecode12.csv` |
| `"X1.0"` | `"US"` | `phecodeX.csv` |
| `"X1.0"` | `"WHO"` | `phecodeX_WHO.csv` |
| `"X"` | — | alias for the latest phecodeX release (currently `X1.0`) |

Use `available_phecode_versions()` to list the canonical versions (`["1.2", "X1.0"]`). The `"X"` alias is a pointer, not a table, so it is not listed. **Pin a release such as `"X1.0"` for reproducibility** — `"X"` changes meaning when a new phecodeX release ships.

Phecode 1.2 has no WHO mapping; requesting that combination raises `ValueError`.

### Returned Columns
With `keep_all_columns=False`:
- phecodeX: `phecode`, `ICD`, `flag`
- phecode 1.2: `phecode_unrolled`, `ICD`, `flag`

With `keep_all_columns=True` (default), all columns of the mapping file are returned — phecode 1.2 additionally includes `exclude_range` and `phecode_unrolled`.

### Raises
`ValueError` if the phecode version, the ICD version, or their combination is unsupported, or if `icd_version="custom"` is requested without `phecode_map_file_path`.

### Notebook Example
```python
from phetk.phecode import get_phecode_map, available_phecode_versions

available_phecode_versions()          # ['1.2', 'X1.0']

phecode_map = get_phecode_map()                          # latest phecodeX, US ICD
phecode_map = get_phecode_map("X1.0")                    # pinned release
phecode_map = get_phecode_map("X", icd_version="WHO")    # WHO ICD-10
phecode_map = get_phecode_map("1.2")                     # phecode 1.2

phecode_map.write_csv("phecode_map.csv")
```

## count_phecode

Generate phecode counts from ICD code data. Maps ICD codes to phecodes and aggregates counts per person-phecode combination.

### Key Parameters
- `phecode_version`: Phecode version to use — `"1.2"`, the alias `"X"` for the latest phecodeX release, or a pinned release such as `"X1.0"` (str, default: `"X"`). Pin a release for reproducibility.
- `icd_version`: ICD mapping version, "US", "WHO", or "custom" (str, default: "US")
- `phecode_map_file_path`: Path to custom phecode mapping table (str, optional)
- `output_file_path`: Path for output TSV file (str, optional)
- `engine`: Backend used for the ICD→phecode join and aggregation, `"duckdb"` or `"polars"` (str, default: `"duckdb"`). `"duckdb"` uses a streaming, spill-capable pipeline with a configurable memory ceiling, which keeps memory bounded on large cohorts. `"polars"` runs fully in-memory and is typically a bit faster when the working set comfortably fits in RAM.
- `memory_limit`: DuckDB memory ceiling, e.g. `"10GB"` (str, optional). Only used when `engine="duckdb"`. Defaults to ~90% of currently available RAM when not provided.

### Notebook Example
```python
from phetk.phecode import Phecode

# All of Us platform
phecode = Phecode(platform="aou")
phecode.count_phecode(
    phecode_version="X",
    icd_version="US",
    output_file_path="phecode_counts.tsv",
    engine="duckdb",
)
```

### CLI Example
```bash
phetk phecode count-phecode \
  --platform "aou" \
  --phecode_version "X" \
  --icd_version "US" \
  --output_file_path "phecode_counts.tsv" \
  --engine "duckdb"
```

## add_age_at_first_event

Calculate age at first phecode event for each participant. Adds age_at_first_event column to phecode counts.

### Key Parameters
- `phecode_count_file_path`: Path to phecode counts TSV file (str, required)
- `output_file_path`: Path for output file with age calculations (str, optional)

### Notebook Example
```python
from phetk.phecode import Phecode

phecode = Phecode(platform="aou")
phecode.add_age_at_first_event(
    phecode_count_file_path="phecode_counts.tsv",
    output_file_path="phecode_counts_with_age.tsv"
)
```

### CLI Example
```bash
phetk phecode add-age-at-first-event \
  --phecode_count_file_path "phecode_counts.tsv" \
  --output_file_path "phecode_counts_with_age.tsv"
```

## add_phecode_time_to_event

Calculate time from study start to first phecode event for survival analysis. Adds phecode_time_to_event column.

### Key Parameters
- `phecode_count_file_path`: Path to phecode counts CSV/TSV file (str, required)
- `cohort_file_path`: Path to cohort file with study start dates (str, required)
- `study_start_date_col`: Column name containing study start dates (str, required)
- `time_unit`: Time unit for calculations, "days" or "years" (str, default: "days")
- `output_file_path`: Path for output file (str, optional)

### Notebook Example
```python
from phetk.phecode import Phecode

# Static method - no instance needed
Phecode.add_phecode_time_to_event(
    phecode_count_file_path="phecode_counts.tsv",
    cohort_file_path="cohort.tsv",
    study_start_date_col="study_start_date",
    time_unit="years",
    output_file_path="phecode_counts_with_time.tsv"
)
```

### CLI Example
```bash
phetk phecode add-phecode-time-to-event \
  --phecode_count_file_path "phecode_counts.tsv" \
  --cohort_file_path "cohort.tsv" \
  --study_start_date_col "study_start_date" \
  --time_unit "years" \
  --output_file_path "phecode_counts_with_time.tsv"
```

## Platform Configuration

### All of Us
```python
phecode = Phecode(platform="aou")
```

- `gbq_dataset_id`: Optional BigQuery dataset ID (overrides `WORKSPACE_CDR`)

### Custom Platform
```python
phecode = Phecode(platform="custom", icd_file_path="path/to/icd_data.tsv")
```

## Input Data Format

### ICD Data Requirements
Required columns for custom ICD data:
- `person_id`: Participant identifier
- `date`: Date of ICD code occurrence
- `ICD`: ICD code value
- `vocabulary_id`: "ICD9CM" or "ICD10CM" (or use `flag` column with values 9/10)

Example:
| person_id | date       | vocabulary_id | ICD   |
|-----------|------------|---------------|-------|
| 13579     | 2010-01-11 | ICD9CM        | 786.2 |
| 13579     | 2017-12-04 | ICD10CM       | R05.1 |

### Custom Phecode Mapping
Required columns for custom phecode mapping:
- `phecode`: Phecode value
- `ICD`: ICD code value
- `flag`: ICD version (9 or 10)
- `sex`: Applicable sex ("Male", "Female", or "Both")
- `phecode_string`: Phecode description
- `phecode_category`: Phecode category
- `exclude_range`: Exclusion range for phecode 1.2

`icd_version="custom"` works with `get_phecode_map` as well, so a custom mapping file can be loaded and inspected before running `count_phecode`:

```python
from phetk.phecode import get_phecode_map

custom_map = get_phecode_map(
    phecode_version="X",
    icd_version="custom",
    phecode_map_file_path="my_phecode_map.csv",
)
```

The file is parsed with the schema of `phecode_version`, so pass the version whose column layout the custom file follows.