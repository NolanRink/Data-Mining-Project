# Florida WQP Script Run Order

This document gives the current script-based run order for the corrected 2021-2023 Florida Water Quality Portal data mining project.

## Required Input Files

The canonical raw data are stored as ZIP files in `data/` and tracked with Git LFS:

- `data/resultphyschem (2).zip`
- `data/station (1).zip`

The active scripts read extracted CSV files:

- `data/resultphyschem.csv`
- `data/station.csv`

If the extracted CSVs are missing, extract them first:

```powershell
git lfs pull
Expand-Archive -LiteralPath "data/resultphyschem (2).zip" -DestinationPath "data" -Force
Expand-Archive -LiteralPath "data/station (1).zip" -DestinationPath "data" -Force
```

Verified current baseline:

- 3,148,368 result rows
- 19,948 station rows
- `ActivityStartDate` from 2021-01-01 through 2023-12-31
- Years present: 2021, 2022, 2023

## Script Command Order

Run from the repository root:

```powershell
python scripts\florida_wqp_eda.py
python scripts\florida_wqp_modeling.py
```

Optional compile checks:

```powershell
python -m py_compile scripts\florida_wqp_eda.py
python -m py_compile scripts\florida_wqp_modeling.py
```

## EDA Script

`scripts/florida_wqp_eda.py` reads:

- `data/resultphyschem.csv`
- `data/station.csv`

It performs raw data profiling, selected-column processing, numeric parsing, unit review, site metadata joins, candidate variable screening, site matrix creation, range screening, and validation.

It writes:

- `outputs/eda/*.csv`
- `outputs/figures/*.png`
- `data/processed/florida_wqp_numeric_long.csv`
- `data/processed/florida_wqp_site_characteristic_matrix.csv`
- `data/processed/florida_wqp_site_characteristic_matrix_metadata.csv`

## Modeling Script

`scripts/florida_wqp_modeling.py` reads the processed and EDA outputs created by the EDA script.

It runs:

- modeling input audit
- selected-variable audit
- site-level Spearman correlation
- PCA
- quality-controlled clustering
- supervised target feasibility screening
- temporal modeling readiness screening
- low-DO supervised readiness checks
- leakage-controlled low-DO supervised classification
- gradient boosting comparison
- threshold sensitivity analysis
- probability-threshold analysis
- permutation importance

The low-DO supervised pass uses a project-defined site-year-season target, excludes leakage predictors, and uses grouped-by-site train/test splitting. It should be presented as exploratory classification, not pollution prediction, regulatory assessment, causal inference, or long-term trend modeling.

The modeling script writes CSV outputs under `outputs/modeling/` and figures under `outputs/figures/modeling/`.

## Runtime Notes

The EDA script reads a large raw WQP result CSV and is the expensive first step. The modeling script can also take time because it includes grouped cross-validation, threshold sensitivity checks, and permutation importance.

Old 2014 wrong-data archive files are not current project evidence.

