# Phecode Module

Phecode module is used to retrieve ICD code data of participants, map ICD codes to phecode 1.2 or phecodeX 1.0, 
and aggregate the counts for each phecode of each participant.

The ICD code retrieval is done automatically for _All of Us_ platform when users instantiate class Phecode.
For other platforms, users must provide your own ICD code data.

## Example ICD code data format

Each row must be unique, i.e., there should not be 2 instances of 1 ICD code in the same day.
Data must have these exact column names. 
"vocabulary_id" column can be replaced with the "flag" column with values of 9 and 10.

### Example with "vocabulary_id" column:

| person_id | date      | vocabulary_id | ICD   |
|-----------|-----------|---------------|-------|
| 13579     | 1-11-2010 | ICD9CM        | 786.2 |
| 13579     | 1-31-2010 | ICD9CM        | 786.2 |
| 13579     | 12-4-2017 | ICD10CM       | R05.1 |
| 24680     | 3-12-2012 | ICD9CM        | 659.2 |
| 24680     | 4-18-2018 | ICD10CM       | R50   |

### Example with "flag" column:

| person_id | date      | flag | ICD   |
|-----------|-----------|------|-------|
| 13579     | 1-11-2010 | 9    | 786.2 |
| 13579     | 1-31-2010 | 9    | 786.2 |
| 13579     | 12-4-2017 | 10   | R05.1 |
| 24680     | 3-12-2012 | 9    | 659.2 |
| 24680     | 4-18-2018 | 10   | R50   |

## Usage Examples

In these examples, we will map US ICD codes (ICD-9-CM & ICD-10-CM) to phecodeX for _All of Us_ and custom platforms.

### Jupyter Notebook example for _All of Us_:
```python
from PheTK.Phecode import Phecode

phecode = Phecode(platform="aou")
phecode.count_phecode(
    phecode_version="X", 
    icd_version="US",
    phecode_map_file_path=None, 
    output_file_name="my_phecode_counts.tsv"
)
```

### Jupyter Notebook example for other platforms
```python
from PheTK.Phecode import Phecode

phecode = Phecode(platform="custom", icd_df_path="/path/to/my_icd_data.tsv")
phecode.count_phecode(
    phecode_version="X", 
    icd_version="US", 
    phecode_map_file_path=None,
    output_file_name="my_phecode_counts.tsv"
)
```

## Custom phecode mapping

Users can provide their own phecode mapping file by adding a csv/tsv file path to `phecode_map_file_path`.
If users provide their own ICD data, the platform should be set to "custom".

Custom phecode mapping file must have columns:
- `phecode`: contains code values of phecodes
- `ICD`: contains code values of ICD codes
- `flag`: specifies whether ICD codes are of version 9 or 10; takes numerical 9 or 10 as values.
- `sex`: sex of ICD/phecode; takes "Male", "Female", or "Both" as values.
- `phecode_string`: literal name of phecodes as strings
- `phecode_category`: specifies which category a phecode belongs to.
- `exclude_range`: only applicable for phecode 1.2.; specifies the range of exclusion if used in PheWAS;
for example, "008.5,008.7,008.51,008.52,008.6"