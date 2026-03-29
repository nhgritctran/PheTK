import datetime
import numpy as np
import polars as pl
import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "aou: mark test as requiring AoU Workbench environment")


# ---------------------------------------------------------------------------
# Synthetic ICD data (mirrors OMOP query output)
# ---------------------------------------------------------------------------

@pytest.fixture
def icd_data():
    return pl.DataFrame({
        "person_id": [1, 1, 2, 2, 3, 3, 4, 5],
        "date": [
            datetime.date(2020, 1, 1), datetime.date(2020, 2, 1),
            datetime.date(2020, 1, 15), datetime.date(2021, 3, 1),
            datetime.date(2019, 6, 1), datetime.date(2020, 8, 1),
            datetime.date(2021, 1, 1), datetime.date(2020, 5, 5),
        ],
        "ICD": ["E11", "I10", "E11", "J45", "E11", "E11", "I10", "J45"],
        "vocabulary_id": ["ICD10CM"] * 8,
    })


@pytest.fixture
def icd_file(icd_data, tmp_path):
    path = tmp_path / "icd_events.tsv"
    icd_data.write_csv(str(path), separator="\t")
    return str(path)


# ---------------------------------------------------------------------------
# Synthetic cohort data
# ---------------------------------------------------------------------------

@pytest.fixture
def cohort_data():
    np.random.seed(42)
    n = 100
    return pl.DataFrame({
        "person_id": list(range(1, n + 1)),
        "age": np.random.randint(18, 80, n).tolist(),
        "sex": np.random.randint(0, 2, n).tolist(),
        "pc1": np.random.uniform(-1, 1, n).tolist(),
        "pc2": np.random.uniform(-1, 1, n).tolist(),
        "pc3": np.random.uniform(-1, 1, n).tolist(),
        "independent_variable_of_interest": np.random.randint(0, 2, n).tolist(),
    })


@pytest.fixture
def cohort_file(cohort_data, tmp_path):
    path = tmp_path / "cohort.tsv"
    cohort_data.write_csv(str(path), separator="\t")
    return str(path)


# ---------------------------------------------------------------------------
# Fake BigQuery response DataFrames (AoU OMOP shape)
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_age_df():
    return pl.DataFrame({
        "person_id": [1, 2, 3],
        "date_of_birth": [
            datetime.date(1970, 1, 1),
            datetime.date(1985, 6, 15),
            datetime.date(1962, 3, 20),
        ],
        "year_of_birth": [1970, 1985, 1962],
        "current_age": [55.0, 40.0, 63.0],
        "current_age_squared": [3025.0, 1600.0, 3969.0],
        "current_age_cubed": [166375.0, 64000.0, 250047.0],
    })


@pytest.fixture
def fake_ehr_df():
    return pl.DataFrame({
        "person_id": [1, 2, 3],
        "last_ehr_date": [
            datetime.date(2023, 1, 1),
            datetime.date(2022, 6, 1),
            datetime.date(2021, 12, 1),
        ],
        "ehr_length": [10.5, 5.2, 8.9],
        "dx_code_occurrence_count": [150, 80, 200],
        "dx_condition_count": [30, 20, 45],
        "age_at_last_ehr_event": [52.0, 38.0, 61.0],
        "age_at_last_ehr_event_squared": [2704.0, 1444.0, 3721.0],
        "age_at_last_ehr_event_cubed": [140608.0, 54872.0, 226981.0],
    })


@pytest.fixture
def fake_sex_df():
    return pl.DataFrame({
        "person_id": [1, 2, 3],
        "sex_at_birth": [1, 0, 1],
    })


# ---------------------------------------------------------------------------
# AoU environment variable patch
# ---------------------------------------------------------------------------

@pytest.fixture
def aou_env(monkeypatch):
    monkeypatch.setenv("WORKSPACE_CDR", "test_project.test_cdr")
    monkeypatch.setenv("GOOGLE_PROJECT", "test-project-id")
    monkeypatch.setenv("WORKSPACE_BUCKET", "gs://test-bucket")
    monkeypatch.setenv("OWNER_EMAIL", "test.user@test.com")
    monkeypatch.setenv("ARTIFACT_REGISTRY_DOCKER_REPO", "us-docker.pkg.dev/test-project/test-repo")
