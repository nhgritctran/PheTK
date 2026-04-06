"""
Unit tests for phewas.py — Cox regression control selection.
"""
import datetime
import polars as pl
import pytest

from phetk.phewas import PheWAS


@pytest.fixture
def cox_phewas(tmp_path):
    """Create a minimal PheWAS instance configured for Cox regression.

    Participants:
      10 — has phecode "ID_001" with first_event_date (2019-01-01) BEFORE
           cox_start_date (2020-01-01) → pre-existing condition
      20 — has phecode "ID_001" with first_event_date (2021-06-01) AFTER
           cox_start_date (2020-01-01) → valid case
      30 — has no phecode "ID_001" → should be a control
    """
    # Cohort file — every participant needs covariates + Cox time columns
    cohort = pl.DataFrame({
        "person_id": [10, 20, 30],
        "independent_variable_of_interest": [1, 0, 1],
        "sex": [1, 0, 1],
        "age": [50, 60, 55],
        "cox_start_date": [
            datetime.date(2020, 1, 1),
            datetime.date(2020, 1, 1),
            datetime.date(2020, 1, 1),
        ],
        "control_observed_time": [5.0, 5.0, 5.0],
    })
    cohort_file = str(tmp_path / "cohort.tsv")
    cohort.write_csv(cohort_file, separator="\t")

    # Phecode counts file
    phecode_counts = pl.DataFrame({
        "person_id": [10, 20],
        "phecode": ["ID_001", "ID_001"],
        "count": [3, 2],
        "first_event_date": [
            datetime.date(2019, 1, 1),   # BEFORE cox_start_date
            datetime.date(2021, 6, 1),   # AFTER  cox_start_date
        ],
        "phecode_observed_time": [1.0, 1.5],
    })
    phecode_file = str(tmp_path / "phecode_counts.tsv")
    phecode_counts.write_csv(phecode_file, separator="\t")

    pw = PheWAS(
        phecode_version="X",
        phecode_count_file_path=phecode_file,
        cohort_file_path=cohort_file,
        covariate_cols=["age", "sex"],
        independent_variable_of_interest="independent_variable_of_interest",
        sex_at_birth_col="sex",
        method="cox",
        cox_start_date_col="cox_start_date",
        cox_control_observed_time_col="control_observed_time",
        cox_phecode_observed_time_col="phecode_observed_time",
        min_cases=1,
        min_phecode_count=2,
        output_file_path=str(tmp_path / "results"),
    )
    return pw


@pytest.fixture
def male_only_phewas(tmp_path):
    """Create a PheWAS instance with an all-male cohort (sex=1).

    Participants 10, 20, 30 are all male. Phecode "MALE_001" has sex
    restriction "Male" and "FEM_001" has sex restriction "Female" in the
    mapping table.
    """
    cohort = pl.DataFrame({
        "person_id": [10, 20, 30],
        "independent_variable_of_interest": [1, 0, 1],
        "sex": [1, 1, 1],
        "age": [50, 60, 55],
    })
    cohort_file = str(tmp_path / "cohort.tsv")
    cohort.write_csv(cohort_file, separator="\t")

    phecode_counts = pl.DataFrame({
        "person_id": [10, 20],
        "phecode": ["FEM_001", "FEM_001"],
        "count": [3, 2],
    })
    phecode_file = str(tmp_path / "phecode_counts.tsv")
    phecode_counts.write_csv(phecode_file, separator="\t")

    pw = PheWAS(
        phecode_version="X",
        phecode_count_file_path=phecode_file,
        cohort_file_path=cohort_file,
        covariate_cols=["age", "sex"],
        independent_variable_of_interest="independent_variable_of_interest",
        sex_at_birth_col="sex",
        min_cases=1,
        min_phecode_count=1,
        output_file_path=str(tmp_path / "results"),
    )
    return pw


@pytest.fixture
def female_only_phewas(tmp_path):
    """Create a PheWAS instance with an all-female cohort (sex=0).

    Participants 10, 20, 30 are all female. Phecode "MALE_001" has sex
    restriction "Male" in the mapping table.
    """
    cohort = pl.DataFrame({
        "person_id": [10, 20, 30],
        "independent_variable_of_interest": [1, 0, 1],
        "sex": [0, 0, 0],
        "age": [50, 60, 55],
    })
    cohort_file = str(tmp_path / "cohort.tsv")
    cohort.write_csv(cohort_file, separator="\t")

    phecode_counts = pl.DataFrame({
        "person_id": [10, 20],
        "phecode": ["MALE_001", "MALE_001"],
        "count": [3, 2],
    })
    phecode_file = str(tmp_path / "phecode_counts.tsv")
    phecode_counts.write_csv(phecode_file, separator="\t")

    pw = PheWAS(
        phecode_version="X",
        phecode_count_file_path=phecode_file,
        cohort_file_path=cohort_file,
        covariate_cols=["age", "sex"],
        independent_variable_of_interest="independent_variable_of_interest",
        sex_at_birth_col="sex",
        min_cases=1,
        min_phecode_count=1,
        output_file_path=str(tmp_path / "results"),
    )
    return pw


class TestSingleSexCohortSexRestriction:
    """Single-sex cohorts must return empty cases/controls for opposite-sex phecodes."""

    def test_male_cohort_female_phecode_returns_empty(self, male_only_phewas):
        """All-male cohort must skip Female-restricted phecodes."""
        cases, controls, _ = male_only_phewas._case_control_prep(
            phecode="FEM_001", keep_ids=True,
        )
        assert len(cases) == 0
        assert len(controls) == 0

    def test_female_cohort_male_phecode_returns_empty(self, female_only_phewas):
        """All-female cohort must skip Male-restricted phecodes."""
        cases, controls, _ = female_only_phewas._case_control_prep(
            phecode="MALE_001", keep_ids=True,
        )
        assert len(cases) == 0
        assert len(controls) == 0


class TestCoxPreExistingExcludedFromControls:
    """Participants with a pre-existing condition (first_event_date < cox_start_date)
    must not appear in the control group."""

    def test_pre_existing_not_in_controls(self, cox_phewas):
        cases, controls, _ = cox_phewas._case_control_prep(
            phecode="ID_001", keep_ids=True,
        )
        control_ids = controls["person_id"].to_list()
        # Person 10 had the condition before cox_start_date — must NOT be a control
        assert 10 not in control_ids

    def test_pre_existing_not_in_cases(self, cox_phewas):
        cases, controls, _ = cox_phewas._case_control_prep(
            phecode="ID_001", keep_ids=True,
        )
        case_ids = cases["person_id"].to_list()
        # Person 10 was excluded pre-baseline — must NOT be a case either
        assert 10 not in case_ids

    def test_valid_case_present(self, cox_phewas):
        cases, controls, _ = cox_phewas._case_control_prep(
            phecode="ID_001", keep_ids=True,
        )
        case_ids = cases["person_id"].to_list()
        assert 20 in case_ids

    def test_clean_participant_is_control(self, cox_phewas):
        cases, controls, _ = cox_phewas._case_control_prep(
            phecode="ID_001", keep_ids=True,
        )
        control_ids = controls["person_id"].to_list()
        assert 30 in control_ids
