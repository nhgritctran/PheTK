"""
Unit tests for _queries.py — validate SQL string structure without executing queries.
All tests are pure string checks; no BigQuery access required.
"""
import pytest
from phetk import _queries


class TestPhecodeIcdQuery:
    def test_returns_string(self):
        result = _queries.phecode_icd_query("my_project.my_cdr")
        assert isinstance(result, str)

    def test_contains_dataset(self):
        ds = "my_project.my_cdr"
        result = _queries.phecode_icd_query(ds)
        assert ds in result

    def test_references_condition_occurrence(self):
        result = _queries.phecode_icd_query("ds")
        assert "condition_occurrence" in result

    def test_references_observation(self):
        result = _queries.phecode_icd_query("ds")
        assert "observation" in result

    def test_references_concept(self):
        result = _queries.phecode_icd_query("ds")
        assert "concept" in result

    def test_references_icd9cm(self):
        result = _queries.phecode_icd_query("ds")
        assert "ICD9CM" in result

    def test_references_icd10cm(self):
        result = _queries.phecode_icd_query("ds")
        assert "ICD10CM" in result

    def test_selects_person_id(self):
        result = _queries.phecode_icd_query("ds")
        assert "person_id" in result

    def test_selects_date(self):
        result = _queries.phecode_icd_query("ds")
        assert "date" in result

    def test_handles_v_codes(self):
        result = _queries.phecode_icd_query("ds")
        assert 'V%' in result
        assert "concept_relationship" in result

    def test_union_distinct(self):
        result = _queries.phecode_icd_query("ds")
        assert "UNION DISTINCT" in result


class TestCurrentAgeQuery:
    def test_returns_string(self):
        result = _queries.current_age_query("ds", (1, 2, 3))
        assert isinstance(result, str)

    def test_contains_dataset(self):
        ds = "my_project.my_cdr"
        result = _queries.current_age_query(ds, (1, 2))
        assert ds in result

    def test_contains_participant_ids(self):
        ids = (100, 200, 300)
        result = _queries.current_age_query("ds", ids)
        assert str(ids) in result

    def test_references_person_table(self):
        result = _queries.current_age_query("ds", (1,))
        assert "person" in result

    def test_selects_current_age(self):
        result = _queries.current_age_query("ds", (1,))
        assert "current_age" in result

    def test_selects_date_of_birth(self):
        result = _queries.current_age_query("ds", (1,))
        assert "date_of_birth" in result

    def test_selects_year_of_birth(self):
        result = _queries.current_age_query("ds", (1,))
        assert "year_of_birth" in result

    def test_handles_death_table(self):
        result = _queries.current_age_query("ds", (1,))
        assert "death" in result


class TestEhrDxCodeQuery:
    def test_returns_string(self):
        result = _queries.ehr_dx_code_query("ds", (1, 2, 3))
        assert isinstance(result, str)

    def test_contains_dataset(self):
        ds = "test.cdr"
        result = _queries.ehr_dx_code_query(ds, (1,))
        assert ds in result

    def test_references_condition_occurrence(self):
        result = _queries.ehr_dx_code_query("ds", (1,))
        assert "condition_occurrence" in result

    def test_selects_ehr_length(self):
        result = _queries.ehr_dx_code_query("ds", (1,))
        assert "ehr_length" in result

    def test_selects_dx_code_occurrence_count(self):
        result = _queries.ehr_dx_code_query("ds", (1,))
        assert "dx_code_occurrence_count" in result

    def test_selects_dx_condition_count(self):
        result = _queries.ehr_dx_code_query("ds", (1,))
        assert "dx_condition_count" in result

    def test_selects_age_at_last_ehr_event(self):
        result = _queries.ehr_dx_code_query("ds", (1,))
        assert "age_at_last_ehr_event" in result

    def test_contains_participant_ids(self):
        ids = (1, 2, 3)
        result = _queries.ehr_dx_code_query("ds", ids)
        assert str(ids) in result


class TestSexAtBirthQuery:
    def test_returns_string(self):
        result = _queries.sex_at_birth("ds", (1, 2, 3))
        assert isinstance(result, str)

    def test_contains_dataset(self):
        ds = "test.cdr"
        result = _queries.sex_at_birth(ds, (1,))
        assert ds in result

    def test_uses_male_concept_id(self):
        result = _queries.sex_at_birth("ds", (1,))
        assert "1585846" in result

    def test_uses_female_concept_id(self):
        result = _queries.sex_at_birth("ds", (1,))
        assert "1585847" in result

    def test_selects_sex_at_birth(self):
        result = _queries.sex_at_birth("ds", (1,))
        assert "sex_at_birth" in result

    def test_references_person_table(self):
        result = _queries.sex_at_birth("ds", (1,))
        assert "person" in result

    def test_male_equals_one(self):
        result = _queries.sex_at_birth("ds", (1,))
        assert "1 AS sex_at_birth" in result

    def test_female_equals_zero(self):
        result = _queries.sex_at_birth("ds", (1,))
        assert "0 AS sex_at_birth" in result
