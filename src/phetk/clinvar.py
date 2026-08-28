"""
Query ClinVar variants by gene symbol or genomic region.

This module reads NCBI's ClinVar VCF release directly over HTTP using tabix
range requests (via ``pysam``), so a whole-gene query transfers only the bytes
for that interval. The VCF is used instead of the E-utilities ``esummary``
endpoint because it is the only access surface that reliably carries rsIDs,
reference/alternate alleles, and allele frequencies, and because filtering on
VCF INFO fields keeps the classification vocabulary under our control -- the
Entrez query parser silently rewrites unknown field tags into free-text
searches, which produces plausible-looking but wrong result sets.

The output frame is shaped to feed straight into ``Cohort.by_genotype()``.

Notes:
    Allele frequencies in ClinVar (``AF_ESP``, ``AF_EXAC``, ``AF_TGP``) are
    single global numbers. ClinVar carries no ancestry-stratified frequencies;
    use gnomAD if ancestry-specific frequencies are required.

    Variants without genomic coordinates are absent from the VCF by
    construction and are therefore not returned.
"""

import atexit
import functools
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request

import polars as pl
# noinspection PyUnresolvedReferences,PyProtectedMember
from phetk import _utils


__all__ = ["ClinVar", "available_clinical_significances",
           "available_review_statuses", "get_gene_region"]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLINVAR_VCF_URL = "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_{assembly}/clinvar.vcf.gz"

_ASSEMBLIES = {
    "grch38": "GRCh38",
    "hg38": "GRCh38",
    "grch37": "GRCh37",
    "hg19": "GRCh37",
}

# ClinVar review status -> star rating shown in the ClinVar web UI. The star
# count is derived from the review status string, never stored in the record.
REVIEW_STATUS_STARS: dict[str, int] = {
    "practice guideline": 4,
    "reviewed by expert panel": 3,
    "criteria provided, multiple submitters, no conflicts": 2,
    "criteria provided, single submitter": 1,
    "criteria provided, conflicting classifications": 1,
    "no assertion criteria provided": 0,
    "no classification provided": 0,
    "no classification for the single variant": 0,
    "no classifications from unflagged records": 0,
}

# Canonical CLNSIG components. Real CLNSIG values are frequently compound
# ("Pathogenic/Likely_pathogenic", "Pathogenic|drug_response"); a record matches
# a requested value when that value is one of the compound's components.
CLINICAL_SIGNIFICANCES: list[str] = [
    "Pathogenic",
    "Likely pathogenic",
    "Uncertain significance",
    "Likely benign",
    "Benign",
    "Conflicting classifications of pathogenicity",
    "drug response",
    "risk factor",
    "Likely risk allele",
    "Uncertain risk allele",
    "protective",
    "Affects",
    "other",
    "not provided",
]

# Convenience aliases for CLNVC values that users habitually shorten.
_VARIANT_TYPE_ALIASES = {"snv": "single nucleotide variant"}

_OUTPUT_SCHEMA: dict[str, pl.DataType] = {
    "variant_id": pl.Utf8,
    "rsid": pl.Utf8,
    # chromosome_number/genomic_position deliberately mirror the parameter
    # names of Cohort.by_genotype(), so a row splats straight into it.
    "chromosome_number": pl.Utf8,
    "genomic_position": pl.Int64,
    "ref_allele": pl.Utf8,
    "alt_allele": pl.Utf8,
    "clinical_significance": pl.Utf8,
    "review_status": pl.Utf8,
    "review_star": pl.Int64,
    "variation_id": pl.Utf8,
    "gene": pl.Utf8,
    "variant_type": pl.Utf8,
    "molecular_consequence": pl.Utf8,
    "condition": pl.Utf8,
    "af_esp": pl.Float64,
    "af_exac": pl.Float64,
    "af_tgp": pl.Float64,
}

_EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# NCBI allows 3 requests/second without an API key; stay comfortably under it.
_EUTILS_MIN_INTERVAL_SECONDS = 0.4

# Remote tabix indexes downloaded this process, keyed by VCF URL. Kept in a
# temporary directory so a stale index is never reused across runs.
_INDEX_CACHE: dict[str, str] = {}
_INDEX_CACHE_DIR: tempfile.TemporaryDirectory | None = None

# NCBI assembly accessions share the GCF_000001405 prefix; the minor version
# separates GRCh37 (<= 25) from GRCh38 (>= 26).
_GRCH_ACCESSION_PREFIX = "GCF_000001405."
_LAST_GRCH37_ACCESSION_MINOR = 25


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _norm_info(value) -> str | None:
    """
    Normalize a raw pysam INFO value into a display string.

    pysam splits ``Number=.`` INFO fields on commas, so a single ClinVar value
    such as ``criteria_provided,_single_submitter`` arrives as the tuple
    ``('criteria_provided', '_single_submitter')``. Re-joining on commas and
    replacing underscores with spaces restores the original value.

    Args:
        value: Raw INFO value from pysam: None, a string, or a tuple/list.

    Returns:
        Normalized string, or None when the field is absent or empty.
    """
    if value is None:
        return None
    if isinstance(value, (tuple, list)):
        if not value:
            return None
        value = ",".join("" if v is None else str(v) for v in value)
    text = str(value).replace("_", " ").strip()
    return text or None


def _significance_components(value: str | None) -> set[str]:
    """
    Split a normalized CLNSIG value into its lower-cased components.

    ClinVar joins co-asserted classifications with "/" and "|", e.g.
    "Pathogenic/Likely pathogenic" or "Pathogenic|drug response". Values with
    no separator, such as "Conflicting classifications of pathogenicity", pass
    through intact.

    Args:
        value: Normalized CLNSIG string, or None.

    Returns:
        Set of lower-cased components, including the full value itself.
    """
    if not value:
        return set()
    components = {value.strip()}
    for chunk in re.split(r"[|/]", value):
        chunk = chunk.strip()
        if chunk:
            components.add(chunk)
    return {component.lower() for component in components}


def _parse_molecular_consequence(value) -> str | None:
    """
    Reduce a ClinVar MC field to a comma-separated list of consequence terms.

    MC entries look like ``SO:0001583|missense_variant``; only the term after
    the pipe is kept.

    Args:
        value: Raw MC INFO value from pysam.

    Returns:
        Comma-separated consequence terms, or None when absent.
    """
    text = _norm_info(value)
    if text is None:
        return None
    terms = []
    for item in text.split(","):
        term = item.split("|")[-1].strip()
        if term and term not in terms:
            terms.append(term)
    return ",".join(terms) or None


def _parse_gene_info(value) -> str | None:
    """
    Reduce a ClinVar GENEINFO field to its gene symbols.

    GENEINFO looks like ``CFTR:1080|LOC111674463:111674463``; the numeric gene
    IDs are stripped. Underscores are preserved because they can occur inside
    gene symbols.

    Args:
        value: Raw GENEINFO INFO value from pysam.

    Returns:
        Pipe-separated gene symbols, or None when absent.
    """
    if value is None:
        return None
    if isinstance(value, (tuple, list)):
        value = ",".join(str(v) for v in value)
    symbols = []
    for item in re.split(r"[|,]", str(value)):
        symbol = item.rsplit(":", 1)[0].strip()
        if symbol and symbol not in symbols:
            symbols.append(symbol)
    return "|".join(symbols) or None


def _parse_region(region: str) -> tuple[str, int, int]:
    """
    Parse a genomic region string into contig and 1-based inclusive bounds.

    Args:
        region: Region string such as "chr7:117480025-117668665" or
            "7:117,480,025-117,668,665". The "chr" prefix and thousands
            separators are optional.

    Returns:
        Tuple of (contig, start, end) with contig stripped of any "chr" prefix.

    Raises:
        ValueError: If the region cannot be parsed or start > end.
    """
    text = str(region).replace(",", "").replace(" ", "")
    match = re.fullmatch(r"(?:chr)?([0-9A-Za-z._]+):(\d+)-(\d+)", text, flags=re.IGNORECASE)
    if match is None:
        raise ValueError(
            f"Could not parse region {region!r}. "
            f"Expected format 'chr7:117480025-117668665' or '7:117480025-117668665'."
        )
    contig, start, end = match.group(1), int(match.group(2)), int(match.group(3))
    if start > end:
        raise ValueError(f"Region start ({start}) is greater than region end ({end}) in {region!r}.")
    return contig, start, end


def _display_locus(chromosome: str, start: int, end: int, assembly: str) -> str:
    """
    Format a region using the contig notation conventional for the assembly.

    GRCh38 loci are conventionally written with a "chr" prefix and GRCh37 loci
    without one, matching how ``Cohort.by_genotype()`` builds its locus strings.
    This is display only; the ``chromosome_number`` output column stays unprefixed
    because ``by_genotype()`` adds the prefix itself.

    Args:
        chromosome: Unprefixed contig name.
        start: 1-based inclusive start position.
        end: 1-based inclusive end position.
        assembly: Canonical assembly name.

    Returns:
        Region string such as "chr7:117480025-117668665".
    """
    prefix = "chr" if assembly == "GRCh38" else ""
    return f"{prefix}{chromosome}:{start}-{end}"


def _as_list(value) -> list[str] | None:
    """
    Coerce a scalar-or-sequence filter argument into a list of strings.

    Args:
        value: None, a string, or an iterable of strings.

    Returns:
        List of strings, or None when value is None.
    """
    if value is None:
        return None
    if isinstance(value, str):
        value = [value]
    return [str(item) for item in value]


def _validate_vocabulary(values: list[str], vocabulary: list[str], label: str) -> set[str]:
    """
    Validate user-supplied filter values against a controlled vocabulary.

    Args:
        values: Requested values.
        vocabulary: Accepted values.
        label: Parameter name used in the error message.

    Returns:
        Set of lower-cased accepted values.

    Raises:
        ValueError: If any requested value is not in the vocabulary.
    """
    lookup = {item.lower(): item for item in vocabulary}
    selected = set()
    for value in values:
        key = str(value).replace("_", " ").strip().lower()
        if key not in lookup:
            raise ValueError(
                f"Unsupported {label} {value!r}. Supported values: {', '.join(vocabulary)}."
            )
        selected.add(key)
    return selected


def _eutils_json(endpoint: str, params: dict, attempts: int = 4) -> dict:
    """
    Call an NCBI E-utilities endpoint and parse the JSON response.

    Requests are spaced out to stay under NCBI's rate limit and retried with
    backoff on HTTP 429, which NCBI returns aggressively for unauthenticated
    clients. Set the NCBI_API_KEY environment variable to raise the limit.

    Args:
        endpoint: Endpoint name, e.g. "esearch" or "esummary".
        params: Query parameters.
        attempts: Number of attempts before giving up.

    Returns:
        Parsed JSON payload.

    Raises:
        ValueError: If the request fails or the response is not valid JSON.
    """
    query = {**params, "retmode": "json", "tool": "PheTK"}
    api_key = os.getenv("NCBI_API_KEY")
    if api_key:
        query["api_key"] = api_key
    url = f"{_EUTILS}/{endpoint}.fcgi?" + urllib.parse.urlencode(query)

    delay = _EUTILS_MIN_INTERVAL_SECONDS
    last_error = None
    for attempt in range(attempts):
        time.sleep(delay)
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                payload = response.read()
        except (urllib.error.URLError, TimeoutError, OSError) as err:
            last_error = err
            delay = min(delay * 4, 8.0)
            continue
        try:
            return json.loads(payload)
        except json.JSONDecodeError as err:
            raise ValueError(f"NCBI E-utilities returned invalid JSON ({endpoint}): {err}") from err

    raise ValueError(f"NCBI E-utilities request failed ({endpoint}): {last_error}")


def _assembly_of_accession(assembly_accession: str) -> str | None:
    """
    Map an NCBI assembly accession to a human assembly name.

    Args:
        assembly_accession: Accession such as "GCF_000001405.40".

    Returns:
        "GRCh38", "GRCh37", or None when the accession is neither.
    """
    if not assembly_accession.startswith(_GRCH_ACCESSION_PREFIX):
        return None
    try:
        minor = int(assembly_accession[len(_GRCH_ACCESSION_PREFIX):])
    except ValueError:
        return None
    return "GRCh37" if minor <= _LAST_GRCH37_ACCESSION_MINOR else "GRCh38"


def _fetch_remote_index(vcf_url: str) -> str:
    """
    Download the tabix index for a remote VCF to a private per-process location.

    htslib otherwise caches a remote index into the current working directory
    under the basename of the remote file. Every ClinVar release is named
    "clinvar.vcf.gz", so the GRCh38 and GRCh37 indexes collide: whichever
    assembly is queried second reads the wrong index and fails with "Invalid
    BGZF header". The same collision happens across weekly releases, since the
    cached index is reused without checking that it still matches the VCF.

    Downloading the index ourselves under a name derived from the full URL
    avoids both collisions, and keeps the working directory clean. The download
    lives in a temporary directory removed when the process exits, so a stale
    index is never reused across runs.

    Args:
        vcf_url: URL of the remote bgzipped VCF.

    Returns:
        Local path of the downloaded tabix index.

    Raises:
        ValueError: If the index cannot be downloaded.
    """
    global _INDEX_CACHE_DIR

    cached = _INDEX_CACHE.get(vcf_url)
    if cached is not None:
        return cached

    cache_dir = _INDEX_CACHE_DIR
    if cache_dir is None:
        cache_dir = tempfile.TemporaryDirectory(prefix="phetk_clinvar_")
        atexit.register(cache_dir.cleanup)
        _INDEX_CACHE_DIR = cache_dir

    index_url = f"{vcf_url}.tbi"
    digest = hashlib.sha256(vcf_url.encode()).hexdigest()[:16]
    index_path = os.path.join(cache_dir.name, f"{digest}.tbi")

    try:
        with urllib.request.urlopen(index_url, timeout=120) as response, \
                open(index_path, "wb") as handle:
            shutil.copyfileobj(response, handle)
    except (urllib.error.URLError, TimeoutError, OSError) as err:
        raise ValueError(f"Could not download tabix index {index_url!r}: {err}") from err

    _INDEX_CACHE[vcf_url] = index_path
    return index_path


def _canonical_assembly(assembly: str) -> str:
    """
    Normalize an assembly name to "GRCh38" or "GRCh37".

    Args:
        assembly: Assembly name, e.g. "GRCh38", "grch37", "hg19".

    Returns:
        Canonical assembly name.

    Raises:
        ValueError: If the assembly is not supported.
    """
    key = str(assembly).strip().lower()
    if key not in _ASSEMBLIES:
        raise ValueError(
            f"Unsupported assembly {assembly!r}. Supported values: GRCh38, GRCh37."
        )
    return _ASSEMBLIES[key]


# ---------------------------------------------------------------------------
# Public module-level functions
# ---------------------------------------------------------------------------

def available_clinical_significances() -> list[str]:
    """
    List clinical significance values accepted by ``ClinVar.search()``.

    Returns:
        List of canonical ClinVar clinical significance components.
    """
    return list(CLINICAL_SIGNIFICANCES)


def available_review_statuses() -> list[str]:
    """
    List review status values accepted by ``ClinVar.search()``.

    Returns:
        List of ClinVar review status strings, ordered from 4 stars to 0 stars.
    """
    return sorted(REVIEW_STATUS_STARS, key=lambda status: (-REVIEW_STATUS_STARS[status], status))


@functools.lru_cache(maxsize=128)
def get_gene_region(gene: str, assembly: str = "GRCh38") -> tuple[str, int, int]:
    """
    Look up the genomic span of a gene symbol via the NCBI Gene database.

    Results are cached in-process, so repeated lookups of the same gene issue a
    single network call.

    Args:
        gene: HGNC gene symbol, e.g. "CFTR".
        assembly: Reference assembly, "GRCh38" or "GRCh37".

    Returns:
        Tuple of (chromosome, start, end) with 1-based inclusive coordinates
        and an unprefixed chromosome name, e.g. ("7", 117480025, 117668665).

    Raises:
        ValueError: If the gene is not found, the NCBI request fails, or the
            gene has no coordinates on the requested assembly.

    Examples:
        >>> from phetk.clinvar import get_gene_region
        >>> get_gene_region("CFTR")  # doctest: +SKIP
        ('7', 117480025, 117668665)
    """
    assembly = _canonical_assembly(assembly)
    symbol = str(gene).strip()
    if not symbol:
        raise ValueError("gene must be a non-empty gene symbol.")

    search = _eutils_json("esearch", {
        "db": "gene",
        "term": f"{symbol}[gene] AND human[orgn] AND alive[prop]",
        "retmax": "5",
    })
    gene_ids = search.get("esearchresult", {}).get("idlist", [])
    if not gene_ids:
        raise ValueError(f"No human gene found in NCBI Gene for symbol {symbol!r}.")

    summary = _eutils_json("esummary", {"db": "gene", "id": gene_ids[0]})
    record = summary.get("result", {}).get(gene_ids[0])
    if not record:
        raise ValueError(f"NCBI Gene returned no summary for {symbol!r} (gene id {gene_ids[0]}).")

    placements = []
    for entry in record.get("genomicinfo") or []:
        placements.append(("GRCh38", entry))
    for entry in record.get("locationhist") or []:
        entry_assembly = _assembly_of_accession(str(entry.get("assemblyaccver", "")))
        if entry_assembly is not None:
            placements.append((entry_assembly, entry))

    for entry_assembly, entry in placements:
        if entry_assembly != assembly:
            continue
        chromosome = str(entry.get("chrloc") or record.get("chromosome") or "").strip()
        if not chromosome:
            continue
        start, stop = int(entry["chrstart"]), int(entry["chrstop"])
        # NCBI reports 0-based coordinates in gene orientation, so a
        # minus-strand gene has chrstart > chrstop.
        low, high = min(start, stop), max(start, stop)
        return chromosome, low + 1, high + 1

    raise ValueError(
        f"NCBI Gene has no {assembly} coordinates for {symbol!r} (gene id {gene_ids[0]})."
    )


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class ClinVar:
    """
    Retrieve ClinVar variants for a gene or genomic region.

    Reads NCBI's ClinVar VCF release with tabix range requests, so only the
    bytes covering the requested interval are transferred. The resulting
    DataFrame carries ``chromosome_number``, ``genomic_position``,
    ``ref_allele`` and ``alt_allele``, named to match the parameters of
    ``Cohort.by_genotype()`` so a row can be passed straight through.
    """

    def __init__(self, assembly: str = "GRCh38", vcf_path: str | None = None):
        """
        Initialize a ClinVar reader.

        Args:
            assembly: Reference assembly, "GRCh38" or "GRCh37". Selects the
                corresponding release directory on the NCBI FTP site and the
                assembly used for gene coordinate lookups.
            vcf_path: Path or URL of a ClinVar VCF to read instead of the NCBI
                FTP release. A tabix index must sit next to it. Required in
                environments with restricted network egress, such as the All of
                Us Researcher Workbench. Local and "gs://" paths are accepted.

        Raises:
            ValueError: If the assembly is not supported.
        """
        self.assembly = _canonical_assembly(assembly)
        self.vcf_path = vcf_path or CLINVAR_VCF_URL.format(assembly=self.assembly)
        self._vcf = None

    def _open_vcf(self):
        """
        Open the ClinVar VCF, reusing the handle across searches.

        Returns:
            An open pysam.VariantFile.

        Raises:
            ValueError: If the VCF or its tabix index cannot be opened.
        """
        if self._vcf is None:
            import pysam

            index_filename = None
            if self.vcf_path.startswith(("http://", "https://", "ftp://")):
                index_filename = _fetch_remote_index(self.vcf_path)

            try:
                self._vcf = pysam.VariantFile(self.vcf_path, index_filename=index_filename)
            except (OSError, ValueError) as err:
                raise ValueError(
                    f"Could not open ClinVar VCF at {self.vcf_path!r}: {err}. "
                    f"If network egress is restricted, download clinvar.vcf.gz and "
                    f"clinvar.vcf.gz.tbi and pass vcf_path."
                ) from err
        return self._vcf

    def _fetch(self, chromosome: str, start: int, end: int):
        """
        Fetch VCF records overlapping a 1-based inclusive interval.

        Args:
            chromosome: Contig name, with or without a "chr" prefix.
            start: 1-based inclusive start position.
            end: 1-based inclusive end position.

        Returns:
            Iterator of pysam VariantRecord objects.

        Raises:
            ValueError: If the contig is not present in the VCF.
        """
        vcf = self._open_vcf()
        contig = re.sub(r"^chr", "", str(chromosome), flags=re.IGNORECASE)
        candidates = [contig, f"chr{contig}"]
        if contig.upper() in ("M", "MT"):
            candidates = ["MT", "M", "chrM", "chrMT"]
        for candidate in candidates:
            try:
                return vcf.fetch(candidate, start - 1, end)
            except (ValueError, KeyError):
                continue
        raise ValueError(
            f"Contig {chromosome!r} is not present in the ClinVar VCF at {self.vcf_path!r}."
        )

    def search(
            self,
            gene: str | None = None,
            region: str | None = None,
            clinical_significance: str | list[str] | None = None,
            review_status: str | list[str] | None = None,
            min_review_star: int | None = None,
            variant_type: str | list[str] | None = None,
            max_allele_frequency: float | None = None,
            output_file_path: str | None = None,
    ) -> pl.DataFrame:
        """
        Search ClinVar for variants in a gene or genomic region.

        Args:
            gene: HGNC gene symbol, e.g. "CFTR". Coordinates are looked up via
                NCBI Gene. Mutually exclusive with region.
            region: Genomic region on the reader's assembly, e.g.
                "chr7:117480025-117668665" or "7:117480025-117668665", with
                1-based inclusive coordinates. Mutually exclusive with gene.
            clinical_significance: One or more ClinVar classifications to keep,
                e.g. "Uncertain significance" or ["Pathogenic",
                "Likely pathogenic"]. Compound ClinVar values are matched by
                component, so "Pathogenic" also keeps
                "Pathogenic/Likely pathogenic". See
                ``available_clinical_significances()``.
            review_status: One or more ClinVar review statuses to keep. See
                ``available_review_statuses()``.
            min_review_star: Minimum ClinVar star rating (0-4). Records whose
                review status has no star rating are dropped when this is set.
                Defaults to None, which keeps 0-star records.
            variant_type: One or more CLNVC values to keep, e.g. "SNV" (alias
                for "single nucleotide variant"), "Deletion", "Duplication",
                "Indel", "Insertion", "Microsatellite", "Inversion".
            max_allele_frequency: Keep only variants whose highest reported
                global allele frequency is at or below this value. Variants
                with no reported frequency are kept. The frequencies are global
                numbers from GO-ESP, ExAC, and 1000 Genomes; ClinVar carries no
                ancestry-stratified frequencies.
            output_file_path: Path for the output TSV file. Defaults to
                "clinvar_{gene or region}_{assembly}.tsv" when omitted.

        Returns:
            polars DataFrame of matching variants, one row per alternate
            allele, with the columns documented in docs/clinvar-module.md.

        Raises:
            ValueError: If neither or both of gene and region are given, if a
                filter value is outside the controlled vocabulary, or if the
                VCF cannot be read.

        Examples:
            >>> from phetk.clinvar import ClinVar  # doctest: +SKIP
            >>> ClinVar().search(gene="CFTR",
            ...                  clinical_significance="Uncertain significance",
            ...                  min_review_star=2)  # doctest: +SKIP
        """
        if (gene is None) == (region is None):
            raise ValueError("Provide exactly one of gene or region.")

        if gene is not None:
            chromosome, start, end = get_gene_region(gene, assembly=self.assembly)
            label = str(gene).strip()
            print(f"Gene {label} maps to "
                  f"{_display_locus(chromosome, start, end, self.assembly)} on {self.assembly}.")
        else:
            chromosome, start, end = _parse_region(region)
            label = f"{chromosome}_{start}_{end}"

        significance_filter = None
        if clinical_significance is not None:
            significance_filter = _validate_vocabulary(
                _as_list(clinical_significance), CLINICAL_SIGNIFICANCES, "clinical_significance"
            )

        review_status_filter = None
        if review_status is not None:
            review_status_filter = _validate_vocabulary(
                _as_list(review_status), list(REVIEW_STATUS_STARS), "review_status"
            )

        if min_review_star is not None:
            min_review_star = int(min_review_star)
            if not 0 <= min_review_star <= 4:
                raise ValueError(f"min_review_star must be between 0 and 4, got {min_review_star}.")

        variant_type_filter = None
        if variant_type is not None:
            variant_type_filter = set()
            for value in _as_list(variant_type):
                key = str(value).replace("_", " ").strip().lower()
                variant_type_filter.add(_VARIANT_TYPE_ALIASES.get(key, key))

        if max_allele_frequency is not None:
            max_allele_frequency = float(max_allele_frequency)

        print(f"Fetching ClinVar variants for "
              f"{_display_locus(chromosome, start, end, self.assembly)}...")
        rows = []
        for record in self._fetch(chromosome, start, end):
            rows.extend(self._record_rows(
                record,
                significance_filter=significance_filter,
                review_status_filter=review_status_filter,
                min_review_star=min_review_star,
                variant_type_filter=variant_type_filter,
                max_allele_frequency=max_allele_frequency,
            ))

        # An explicit schema keeps an empty result a correctly typed zero-row
        # frame instead of a schema-inference error.
        df = pl.DataFrame(rows, schema=_OUTPUT_SCHEMA)
        print(f"Found {len(df)} variant(s) matching the requested filters.")

        if output_file_path is None:
            output_file_path = f"clinvar_{re.sub(r'[^0-9A-Za-z._-]', '_', label)}_{self.assembly}.tsv"
        _utils.write_tsv(df, output_file_path)
        print(f"Saved to\033[1m {output_file_path}\033[0m.")
        print()

        return df

    @staticmethod
    def _record_rows(
            record,
            significance_filter: set[str] | None,
            review_status_filter: set[str] | None,
            min_review_star: int | None,
            variant_type_filter: set[str] | None,
            max_allele_frequency: float | None,
    ) -> list[dict]:
        """
        Convert one VCF record into zero or more output rows.

        Multi-allelic records yield one row per alternate allele. Filters are
        applied here so non-matching records never materialize.

        Args:
            record: pysam VariantRecord.
            significance_filter: Lower-cased clinical significance components
                to keep, or None to keep all.
            review_status_filter: Lower-cased review statuses to keep, or None.
            min_review_star: Minimum star rating, or None.
            variant_type_filter: Lower-cased CLNVC values to keep, or None.
            max_allele_frequency: Maximum global allele frequency, or None.

        Returns:
            List of row dicts matching the output schema.
        """
        info = record.info

        significance = _norm_info(info.get("CLNSIG"))
        if significance_filter is not None:
            if not (_significance_components(significance) & significance_filter):
                return []

        status = _norm_info(info.get("CLNREVSTAT"))
        status_key = status.lower() if status else None
        if review_status_filter is not None and status_key not in review_status_filter:
            return []

        star = REVIEW_STATUS_STARS.get(status_key) if status_key else None
        if min_review_star is not None and (star is None or star < min_review_star):
            return []

        vc = _norm_info(info.get("CLNVC"))
        if variant_type_filter is not None:
            if vc is None or vc.lower() not in variant_type_filter:
                return []

        frequencies = []
        for field in ("AF_ESP", "AF_EXAC", "AF_TGP"):
            value = info.get(field)
            if isinstance(value, (tuple, list)):
                value = value[0] if value else None
            frequencies.append(None if value is None else float(value))
        if max_allele_frequency is not None:
            observed = [f for f in frequencies if f is not None]
            if observed and max(observed) > max_allele_frequency:
                return []

        rsid = _norm_info(info.get("RS"))
        rsid = f"rs{rsid}" if rsid else None
        chromosome = re.sub(r"^chr", "", str(record.chrom), flags=re.IGNORECASE)
        gene_symbols = _parse_gene_info(info.get("GENEINFO"))
        consequence = _parse_molecular_consequence(info.get("MC"))
        condition = _norm_info(info.get("CLNDN"))

        rows = []
        for alt in record.alts or (None,):
            rows.append({
                "variant_id": f"{chromosome}:{record.pos}:{record.ref}:{alt}",
                "rsid": rsid,
                "chromosome_number": chromosome,
                "genomic_position": int(record.pos),
                "ref_allele": record.ref,
                "alt_allele": alt,
                "clinical_significance": significance,
                "review_status": status,
                "review_star": star,
                "variation_id": None if record.id is None else str(record.id),
                "gene": gene_symbols,
                "variant_type": vc,
                "molecular_consequence": consequence,
                "condition": condition,
                "af_esp": frequencies[0],
                "af_exac": frequencies[1],
                "af_tgp": frequencies[2],
            })
        return rows


# ---------------------------------------------------------------------------
# CLI entry points
# ---------------------------------------------------------------------------

def main_search():
    """Main entry point for the clinvar search CLI command."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Query ClinVar variants by gene symbol or genomic region"
    )

    parser.add_argument("--gene", "-g", type=str, default=None,
                        help="Gene symbol, e.g. CFTR (mutually exclusive with --region)")
    parser.add_argument("--region", "-r", type=str, default=None,
                        help="Genomic region, e.g. chr7:117480025-117668665, 1-based inclusive "
                             "(mutually exclusive with --gene)")
    parser.add_argument("--assembly", type=str, default="GRCh38",
                        help="Reference assembly: 'GRCh38' or 'GRCh37' (default: GRCh38)")
    parser.add_argument("--vcf_path", type=str, default=None,
                        help="Path or URL of a ClinVar VCF to use instead of the NCBI FTP "
                             "release. Required where network egress is restricted.")

    parser.add_argument("--clinical_significance", "-s", type=str, nargs="+", default=None,
                        help="Clinical significance value(s) to keep, "
                             "e.g. 'Uncertain significance' Pathogenic")
    parser.add_argument("--review_status", type=str, nargs="+", default=None,
                        help="ClinVar review status value(s) to keep")
    parser.add_argument("--min_review_star", type=int, default=None,
                        help="Minimum ClinVar star rating, 0-4 (default: no minimum)")
    parser.add_argument("--variant_type", "-t", type=str, nargs="+", default=None,
                        help="CLNVC variant type(s) to keep, e.g. SNV Deletion")
    parser.add_argument("--max_allele_frequency", type=float, default=None,
                        help="Maximum global allele frequency (GO-ESP/ExAC/1000 Genomes)")

    parser.add_argument("--output_file_path", "-o", type=str, default=None,
                        help="Path for output TSV file")

    args = parser.parse_args()

    try:
        clinvar = ClinVar(assembly=args.assembly, vcf_path=args.vcf_path)
        clinvar.search(
            gene=args.gene,
            region=args.region,
            clinical_significance=args.clinical_significance,
            review_status=args.review_status,
            min_review_star=args.min_review_star,
            variant_type=args.variant_type,
            max_allele_frequency=args.max_allele_frequency,
            output_file_path=args.output_file_path,
        )
    except ValueError as err:
        print(err)
        sys.exit(1)
