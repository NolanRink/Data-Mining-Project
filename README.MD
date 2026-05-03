# Florida Water Quality Data Mining Project

## Project Overview

This repository contains a data mining final project using public Florida Water Quality Portal (WQP) data from 2021-2023. The project studies statewide water-quality patterns and builds an exploratory low dissolved oxygen (DO) classification model from structured/tabular environmental monitoring data.

Water quality and dissolved oxygen are relevant to environmental monitoring, landscape health, and conservation science because chemical and physical water conditions affect aquatic habitat and ecosystem stress. The project follows the full data mining workflow: data acquisition, preprocessing, exploratory analysis, correlation analysis, PCA, clustering, supervised modeling, visualization, interpretation, and comparison with related water-quality machine-learning work.

## Research Questions

1. What water-quality characteristics are most common and useful in the dataset?
2. Which monitoring sites have similar water-quality profiles?
3. Can low dissolved oxygen risk be predicted from other measured water-quality variables?
4. Which variables are most important for predicting low dissolved oxygen?

## Dataset

The project uses Florida WQP physical and chemical result data from 2021-2023. The data are public, structured/tabular, medium-to-large for a class project, and do not require a GPU.

Expected raw ZIP files in `data/`, tracked with Git LFS:

- `data/resultphyschem (2).zip`
- `data/station (1).zip`

The active scripts read the extracted raw CSV files:

- `data/resultphyschem.csv`
- `data/station.csv`


Verified current dataset profile:

- Result rows: 3,148,368
- Station rows: 19,948
- Date range: 2021-01-01 through 2023-12-31
- Years present: 2021, 2022, and 2023
- Unique characteristics: 168
- Duplicate `ResultIdentifier` rows: 0

The old 2014-only dataset was an incorrect earlier download and should not be used as current project evidence.

## Methods

Main methods:

- Data acquisition and preprocessing: Git LFS ZIP handling, raw CSV extraction, date parsing, numeric parsing, unit review, station joins, validation summaries, and site-characteristic matrix creation.
- Exploratory data analysis: characteristic coverage, missingness, site coverage, temporal summaries, unit consistency, and report-ready figures.
- Correlation analysis: site-level Spearman correlations.
- PCA: standardized site-level feature matrices for summarizing major water-quality variation.
- Clustering: k-means clustering for exploratory site-profile grouping.
- Supervised classification: majority baseline, Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting.
- Model evaluation: accuracy, balanced accuracy, precision, recall, F1, ROC-AUC, PR-AUC, confusion matrices, and grouped cross-validation.
- Threshold and probability analysis: low-DO sensitivity at <5, <4, and <3 mg/L, plus Random Forest probability-threshold tradeoffs.
- Permutation importance: scikit-learn permutation importance on the grouped held-out test set.

The supervised target is a project-defined low-DO screening condition: site-year-season median dissolved oxygen <5 mg/L. This is not a regulatory assessment, pollution prediction, causal inference, future forecast, or long-term trend model.

## Main Results

EDA confirmed that the current dataset is broad and uneven, which is typical of public monitoring data. High-volume useful variables include water temperature, pH, dissolved oxygen, phosphorus, dissolved oxygen saturation, turbidity, ammonia, nitrate plus nitrite, and orthophosphate. The site-characteristic matrix contains 18,339 site rows and 21 columns, and EDA selected 48 candidate modeling variables.

PCA and clustering were used as exploratory site-profile methods. Core PCA on pH, water temperature, and dissolved oxygen found that PC1 explained 42.2% of variance and PC2 explained 39.2%, for a combined 81.4%. K-means clustering selected k=4 with silhouette = 0.316. This supports exploratory site-profile grouping, but clusters should not be described as environmental regions or causal groups.

The final supervised presentation model is **Random Forest**. The supervised workflow used a grouped-by-site train/test split with 0 train/test site overlap and excluded dissolved oxygen, dissolved oxygen saturation, ID fields, and target-derived fields from predictors.

Random Forest grouped held-out test metrics:

- Accuracy: 0.752
- Balanced accuracy: 0.760
- Precision: 0.600
- Recall: 0.785
- F1: 0.680
- ROC-AUC: 0.852
- PR-AUC: 0.741

Grouped cross-validation supported the result:

- Mean F1: 0.702
- Mean PR-AUC: 0.763

Gradient Boosting was tested as a literature-aligned boosted-tree comparison model. It improved held-out precision compared with Random Forest, 0.695 versus 0.600, but reduced recall, F1, ROC-AUC, PR-AUC, and grouped-CV F1. Gradient Boosting test F1 was 0.659, ROC-AUC was 0.848, PR-AUC was 0.737, and grouped-CV F1 was 0.666. Therefore, Random Forest remains the final model and Gradient Boosting remains a comparison model.

Threshold sensitivity supported keeping the <5 mg/L low-DO target as the main project-defined screening threshold. Random Forest F1 and PR-AUC were lower for stricter thresholds: <4 mg/L had F1 = 0.599 and PR-AUC = 0.651, while <3 mg/L had F1 = 0.510 and PR-AUC = 0.591.

Permutation importance identified pH, ammonia, water temperature, turbidity, orthophosphate, and phosphorus as the top Random Forest behavior features. These importances describe model behavior on held-out data, not environmental causality.

The project comparison with existing or previously reported findings is included in `reports/wqp_2021_2023_modeling_literature_comparison.md` and summarized in `reports/final_ieee_project_report.md`. The comparison is cautious because published Water Quality Index and DO/hypoxia studies often use different targets, datasets, predictors, and validation splits.

## Reproducibility Workflow

The current reproduction workflow is script-based. It runs on a standard laptop or desktop and does not require a GPU.

Run from Windows PowerShell:

```powershell
git clone <repo-url>
cd "Data Mining Project"
git lfs install
git lfs pull
```

If the extracted CSVs are not already present, extract them from the Git LFS ZIP files:

```powershell
Expand-Archive -LiteralPath "data/resultphyschem (2).zip" -DestinationPath "data" -Force
Expand-Archive -LiteralPath "data/station (1).zip" -DestinationPath "data" -Force
```

Run the project scripts in this order:

```powershell
python scripts\florida_wqp_eda.py
python scripts\florida_wqp_modeling.py
```

`scripts\florida_wqp_eda.py` reads:

- `data/resultphyschem.csv`
- `data/station.csv`

It creates processed files, EDA tables, validation tables, and EDA figures under:

- `data/processed/`
- `outputs/eda/`
- `outputs/figures/`

`scripts\florida_wqp_modeling.py` reads the processed and EDA outputs, then creates correlation, PCA, clustering, supervised modeling, threshold sensitivity, probability-threshold, permutation-importance, metric, and figure outputs under:

- `outputs/modeling/`
- `outputs/figures/modeling/`

The scripts print concise summaries to the console and provide the active reproducibility workflow.

## Expected Outputs

Generated output locations:

- `data/processed/`: regenerated processed datasets, including the numeric long file and site-characteristic matrix.
- `outputs/eda/`: EDA validation tables, characteristic summaries, site summaries, candidate variables, and matrix validation outputs.
- `outputs/modeling/`: PCA, clustering, supervised metrics, cross-validation metrics, confusion matrices, threshold sensitivity, probability-threshold analysis, and permutation importance tables.
- `outputs/figures/`: EDA figures such as record counts, top characteristics, site types, and unit consistency.
- `outputs/figures/modeling/`: modeling figures such as PCA, clustering, model comparison, confusion matrix, feature importance, threshold sensitivity, and precision-recall threshold plots.
- `reports/`: written report drafts, project reviews, final presentation planning, and submission guidance.

Key final report files:

- `reports/final_ieee_project_report.md`
- `reports/submission_package_recommendation.md`
- `reports/wqp_2021_2023_modeling_results.md`
- `reports/wqp_2021_2023_supervised_modeling_report.md`
- `reports/wqp_2021_2023_modeling_literature_comparison.md`
- `reports/wqp_2021_2023_final_presentation_plan.md`

## Repository Organization

- `data/`: raw ZIP files through Git LFS, local extracted CSVs, and regenerated processed data.
- `scripts/`: script-based EDA and modeling workflows.
- `outputs/`: generated EDA/modeling tables and figures.
- `reports/`: final report draft, project reports, review notes, reproducibility notes, and submission guidance.
- Root config/docs: `.gitignore`, `.gitattributes`, `AGENTS.md`, `README.MD`, and supporting project files.
