# Florida WQP Data Mining Project

This repository analyzes Florida Water Quality Portal (WQP) water-quality data for a data mining class project. The project is currently being restarted from the corrected 2021-2023 Florida WQP export.

## Dataset

Data source: Water Quality Portal, https://www.waterqualitydata.us/

Current verified local input files:

- `data/resultphyschem.csv`
- `data/station.csv`

Verified baseline:

- `data/resultphyschem.csv`: 3,148,368 result rows
- `data/station.csv`: 19,948 station rows
- `ActivityStartDate`: 2021-01-01 through 2023-12-31
- Years present: 2021, 2022, 2023
- `ResultIdentifier` is unique
- All result site IDs match `station.csv`

The previous 2014-only dataset was downloaded incorrectly. Old reports, figures, modeling results, and the old presentation are obsolete and should not be cited as current evidence.

## Current Workflow

Run EDA first, then the current 2021-2023 modeling script:

```powershell
python -m py_compile scripts\florida_wqp_eda.py
python scripts\florida_wqp_eda.py
python -m py_compile scripts\florida_wqp_modeling.py
python scripts\florida_wqp_modeling.py
```

The EDA script writes fresh outputs to:

- `outputs/eda/`
- `outputs/figures/`
- `data/processed/`
- `reports/wqp_data_understanding_summary.md`
- `reports/wqp_data_mining_plan.md`

The modeling script now runs the current 2021-2023 modeling workflow:

- modeling input and selected-variable audits
- site-level Spearman correlation
- PCA
- quality-controlled clustering
- supervised target feasibility screening
- temporal modeling readiness screening
- low-DO supervised readiness checks
- first leakage-controlled low-DO supervised classification pass

The supervised pass uses a project-defined low-DO target, grouped-by-site splitting, and leakage exclusions for dissolved oxygen, dissolved oxygen saturation, and ID fields. Results are exploratory and should not be described as pollution prediction, regulatory assessment, causal inference, or long-term trend modeling.

## Current Reports

Reset and planning reports:

- `reports/wqp_new_data_reset_review.md`
- `reports/wqp_new_data_refactor_plan.md`
- `reports/wqp_cleanup_old_2014_outputs_plan.md`
- `reports/wqp_2021_2023_eda_reset_implementation_report.md`

Fresh EDA reports, after running the EDA script:

- `reports/wqp_data_understanding_summary.md`
- `reports/wqp_data_mining_plan.md`

Current modeling and final communication reports:

- `reports/wqp_2021_2023_modeling_results.md`
- `reports/wqp_2021_2023_supervised_modeling_report.md`
- `reports/wqp_2021_2023_supervised_modeling_review.md`
- `reports/wqp_2021_2023_final_report_outline.md`
- `reports/wqp_2021_2023_final_presentation_plan.md`

Old 2014-only archive folders have been deleted from the active repository. They are not current evidence and should not be cited.

## Requirements

Use Python with:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

No new dependencies are required for the EDA reset.

## Key Limitations

- WQP sampling is observational and uneven by provider, site, site type, location, season, and characteristic.
- Three years of data can support data understanding, seasonal summaries, and possible inter-year comparisons, but trend claims require site-characteristic continuity checks.
- Raw values should not be compared across characteristics without unit review and standardization.
- Non-detects, censored values, missing units, and suspicious numeric values need explicit QA handling.
- Supervised low-DO modeling is exploratory and uses a project-defined threshold with grouped-by-site validation.

## Cleanup Status

Old 2014 wrong-data archives and legacy file folders have been removed from the active repository. Use `reports/wqp_delete_2014_old_files_report.md` for the cleanup record.
