# ClinVar module

The `clinvar` module retrieves ClinVar variants for a gene symbol or a genomic region and
returns them as a polars DataFrame whose columns feed directly into
[`Cohort.by_genotype()`](cohort-module.md).

It reads NCBI's ClinVar VCF release over HTTP using tabix range requests, so a whole-gene
query transfers only the bytes covering that interval — a full CFTR fetch takes well under a
second. No new dependency is required; `pysam` already ships with PheTK.

**Why the VCF and not E-utilities?** The `esummary` endpoint is the documented access route,
but it returns rsIDs for only ~3% of records, never returns allele frequencies, and leaves
ref/alt empty on most records. The VCF carries all three. E-utilities also fails silently:
Entrez rewrites unknown field tags such as `[Germline classification]` into `[All Fields]` and
free-text matches, producing plausible-looking but wrong result sets. Filtering on VCF INFO
fields keeps the vocabulary under PheTK's control, so an unrecognized filter value raises
instead of quietly returning the wrong rows.

---

## Quick start

```python
from phetk.clinvar import ClinVar

clinvar = ClinVar()

# All variants of uncertain significance in CFTR, 2 stars or better
vus = clinvar.search(
    gene="CFTR",
    clinical_significance="Uncertain significance",
    min_review_star=2,
)
```

```bash
phetk clinvar search --gene CFTR \
  --clinical_significance "Uncertain significance" \
  --min_review_star 2 \
  -o cftr_vus_2star.tsv
```

---

## `ClinVar(assembly="GRCh38", vcf_path=None)`

| Parameter | Description |
|---|---|
| `assembly` | `"GRCh38"` (default) or `"GRCh37"`. `"hg38"` / `"hg19"` are accepted aliases. Selects the release directory on the NCBI FTP site and the assembly used for gene coordinate lookups. |
| `vcf_path` | Path or URL of a ClinVar VCF to read instead of the NCBI FTP release. A tabix index (`.tbi`) must sit next to it. Local and `gs://` paths are accepted. |

### `vcf_path` on the All of Us Researcher Workbench

The Workbench restricts outbound network egress, so `ftp.ncbi.nlm.nih.gov` may be
unreachable. Stage the release into your workspace bucket first, then point `vcf_path` at it:

```bash
# from a machine with internet access, or a Workbench terminal if egress is permitted
curl -O https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz
curl -O https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz.tbi
gsutil cp clinvar.vcf.gz clinvar.vcf.gz.tbi "${WORKSPACE_BUCKET}/clinvar/"
```

```python
clinvar = ClinVar(vcf_path=f"{os.getenv('WORKSPACE_BUCKET')}/clinvar/clinvar.vcf.gz")

# gene lookup also needs egress, so pass an explicit region instead
variants = clinvar.search(region="chr7:117480025-117668665",
                          clinical_significance="Pathogenic")
```

Note that `gene=` additionally requires access to `eutils.ncbi.nlm.nih.gov` for the coordinate
lookup. Use `region=` when only the VCF has been staged locally.

---

## `ClinVar.search(...)`

| Parameter | Description |
|---|---|
| `gene` | HGNC gene symbol, e.g. `"CFTR"`. Coordinates are looked up via NCBI Gene. Mutually exclusive with `region`. |
| `region` | Genomic region on the reader's assembly, e.g. `"chr7:117480025-117668665"` or `"7:117480025-117668665"`. **1-based inclusive.** The `chr` prefix and thousands separators are optional. Mutually exclusive with `gene`. |
| `clinical_significance` | One or more classifications to keep. Compound ClinVar values are matched by component (see below). |
| `review_status` | One or more ClinVar review statuses to keep. |
| `min_review_star` | Minimum ClinVar star rating, 0–4. Defaults to `None`, which keeps 0-star records. When set, records whose review status has no star rating are dropped. |
| `variant_type` | One or more `CLNVC` values to keep, e.g. `"SNV"`, `"Deletion"`. |
| `max_allele_frequency` | Keep only variants whose highest reported global allele frequency is at or below this value. Variants with no reported frequency are kept. |
| `output_file_path` | Path for the output TSV. Defaults to `clinvar_{gene or region}_{assembly}.tsv`. |

Exactly one of `gene` or `region` is required; passing both or neither raises `ValueError`.

---

## Output columns

| Column | Source | Notes |
|---|---|---|
| `variant_id` | derived | `7:117480082:C:A` |
| `rsid` | `RS` | `rs902914688`, or **null** — about 31% of CFTR VUS have no rsID |
| `chromosome_number` | `CHROM` | **string**, no `chr` prefix, so `X` and `Y` are representable |
| `genomic_position` | `POS` | int, 1-based, on `assembly` |
| `ref_allele` / `alt_allele` | `REF` / `ALT` | direct from the VCF |
| `clinical_significance` | `CLNSIG` | normalized, may be compound |
| `review_status` | `CLNREVSTAT` | normalized |
| `review_star` | derived | 0–4, or null when the status has no rating |
| `variation_id` | `ID` | ClinVar VariationID |
| `gene` | `GENEINFO` | pipe-separated symbols, `:geneid` stripped |
| `variant_type` | `CLNVC` | `single nucleotide variant`, `Deletion`, … |
| `molecular_consequence` | `MC` | comma-separated terms, SO accessions stripped |
| `condition` | `CLNDN` | disease name(s) |
| `af_esp`, `af_exac`, `af_tgp` | `AF_ESP` / `AF_EXAC` / `AF_TGP` | float or null; **global, not ancestry-specific** |

Multi-allelic records are split into one row per alternate allele. An empty result is returned
as a correctly typed zero-row DataFrame rather than an error.

### Variants without coordinates are excluded

ClinVar contains records with no genomic placement (some structural and legacy submissions).
These are absent from the VCF by construction and are therefore never returned. For CFTR this
accounts for the difference between the 2442 VUS returned here and the ~2569 reported by
E-utilities.

---

## Allele frequency is global, not ancestry-specific

`af_esp`, `af_exac`, and `af_tgp` are single global numbers from GO-ESP, ExAC, and 1000
Genomes. **ClinVar carries no ancestry-stratified allele frequencies.** They are useful as a
filtering aid — dropping variants too common to plausibly be pathogenic — but not for
ancestry-aware analysis:

```python
rare_vus = clinvar.search(gene="CFTR",
                          clinical_significance="Uncertain significance",
                          max_allele_frequency=0.001)
```

If you need ancestry-specific frequencies, use gnomAD. That integration is out of scope for
this module.

---

## Clinical significance vocabulary

Accepted values, from `available_clinical_significances()`:

```
Pathogenic                                    drug response
Likely pathogenic                             risk factor
Uncertain significance                        Likely risk allele
Likely benign                                 Uncertain risk allele
Benign                                        protective
Conflicting classifications of pathogenicity  Affects
                                              other
                                              not provided
```

### Compound values are matched by component

ClinVar frequently co-asserts classifications, joining them with `/` or `|`:
`Pathogenic/Likely pathogenic`, `Benign/Likely benign`, `Pathogenic|drug response`,
`Uncertain significance/Uncertain risk allele`.

`search()` splits these into components, so:

```python
clinvar.search(gene="CFTR", clinical_significance="Pathogenic")
```

returns records classified `Pathogenic`, `Pathogenic/Likely pathogenic`, **and**
`Pathogenic|drug response`.

`Conflicting classifications of pathogenicity` contains no separator and is matched intact —
it is *not* returned by a `"Pathogenic"` query.

Values outside the vocabulary raise `ValueError` rather than silently returning nothing.

---

## Review status and star ratings

The star count shown in the ClinVar web UI is derived from the review status string, not
stored in the record. `search()` reconstructs it:

| Review status | Stars |
|---|---|
| `practice guideline` | 4 |
| `reviewed by expert panel` | 3 |
| `criteria provided, multiple submitters, no conflicts` | 2 |
| `criteria provided, single submitter` | 1 |
| `criteria provided, conflicting classifications` | 1 |
| `no assertion criteria provided` | 0 |
| `no classification provided` | 0 |
| `no classification for the single variant` | 0 |
| `no classifications from unflagged records` | 0 |

`min_review_star` is usually the friendlier filter; `review_status` is available when you need
an exact status. Both are listed by `available_review_statuses()`.

**0-star records are kept by default.** `min_review_star` defaults to `None`, so nothing is
dropped on review quality unless you ask for it.

---

## Handing off to `Cohort.by_genotype()`

Four output columns are **named after `by_genotype()`'s parameters** — `chromosome_number`,
`genomic_position`, `ref_allele`, `alt_allele` — so a row splats straight in with no mapping
layer:

```python
from phetk.clinvar import ClinVar
from phetk.cohort import Cohort

variants = ClinVar().search(
    gene="CFTR",
    clinical_significance=["Pathogenic", "Likely pathogenic"],
    min_review_star=2,
)

cohort = Cohort(platform="aou", aou_db_version=8)
locus_columns = ["chromosome_number", "genomic_position", "ref_allele", "alt_allele"]

for variant in variants.select(locus_columns).iter_rows(named=True):
    cohort.by_genotype(**variant, reference_genome="GRCh38")
```

`tests/unit/test_clinvar_unit.py::TestByGenotypeHandoff` checks these names against
`inspect.signature(Cohort.by_genotype)`, so the two cannot drift apart silently.

### Why `chromosome_number` has no `chr` prefix

ClinVar VCF contigs are unprefixed (`7`, `X`, `MT`), and `by_genotype()` **adds the prefix
itself** on GRCh38 (`cohort.py:206`):

```python
base_locus = f"{chromosome_number}:{genomic_position}"
if reference_genome == "GRCh38":
    locus = "chr" + base_locus
```

So a `"chr7"` value here would produce `chrchr7:117480099`. The column stays bare on purpose.
Console messages *do* use the assembly's conventional notation (`chr7:…` on GRCh38), but that
is display only.

### X and Y chromosomes

`chromosome_number` is deliberately a **string** column so `X` and `Y` survive. Despite its
`chromosome_number: int` annotation, `by_genotype()` never uses the value arithmetically — it
only interpolates it into locus strings — so `"7"`, `"X"`, and `"Y"` all pass through the
**Python API** unchanged.

The **CLI is the exception**: `phetk cohort by-genotype --chromosome_number` is declared
`type=int`, so X and Y cannot be passed there. Use the Python API for sex chromosomes.

### Reading the TSV back

polars infers `chromosome_number` as an integer when every value is numeric, which would turn
a mixed autosome/X result into a mixed-type column. Pin it when re-reading:

```python
df = pl.read_csv("clinvar_CFTR_GRCh38.tsv", separator="\t",
                 schema_overrides={"chromosome_number": pl.Utf8})
```

---

## Gene coordinate lookup

```python
from phetk.clinvar import get_gene_region

get_gene_region("CFTR")                     # ('7', 117480025, 117668665)
get_gene_region("CFTR", assembly="GRCh37")  # ('7', 117120079, 117308719)
```

Returns 1-based inclusive coordinates with an unprefixed chromosome name, resolved through
NCBI Gene `esearch` + `esummary`. Results are cached in-process, so repeated lookups of the
same gene cost one round trip total. Minus-strand genes are returned in ascending coordinate
order.

Queries are throttled to stay under NCBI's rate limit (3 requests/second unauthenticated) and
retried with backoff on HTTP 429. Set the `NCBI_API_KEY` environment variable to raise that
limit.

The returned span is the gene's genomic footprint, so a `gene=` search returns every ClinVar
record overlapping that interval — including records annotated to overlapping genes. Check the
`gene` column if you need to restrict further.

---

## Switching assemblies

Every ClinVar release — GRCh38 and GRCh37 alike — is published as `clinvar.vcf.gz`. Left to
its own devices, htslib caches a remote tabix index into the current working directory under
that basename and reuses it without checking that it still matches the VCF, so querying one
assembly and then the other in the same directory reads the wrong index and fails with
`Invalid BGZF header`. The same collision occurs across weekly releases.

PheTK downloads the index itself to a private per-process location keyed by the full URL, so
switching assemblies works and nothing is written to your working directory:

```python
grch38 = ClinVar().search(gene="CFTR", clinical_significance="Pathogenic")
grch37 = ClinVar(assembly="GRCh37").search(gene="CFTR", clinical_significance="Pathogenic")
```

The two frames describe the same variants at different coordinates:

| Assembly | CFTR span | First VUS (2★) |
|---|---|---|
| GRCh38 | 117,480,025–117,668,665 | `7:117480099:A:C` |
| GRCh37 | 117,120,079–117,308,719 | `7:117120153:A:C` |

Match the assembly to your genotype data. All of Us CDR v7/v8 genomic data is **GRCh38**,
which is why it is the default here and in `Cohort.by_genotype(reference_genome=...)`.

---

## Data freshness

The ClinVar VCF is a **weekly snapshot**, so counts will drift from the live ClinVar web UI
between releases. Write assertions against ranges, not exact counts.

---

## Errors

Following the convention used elsewhere in PheTK, the Python API raises `ValueError`, while
the CLI wrapper catches it, prints the message, and exits with status 1.

```python
ClinVar().search(gene="CFTR", region="7:1-100")   # ValueError: Provide exactly one of gene or region.
ClinVar().search(gene="CFTR", clinical_significance="VUS-high")  # ValueError: Unsupported clinical_significance
ClinVar(assembly="GRCh36")                        # ValueError: Unsupported assembly
```
