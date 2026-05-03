# Florida WQP 2021-2023 Data Mining Project

This repository analyzes public Florida Water Quality Portal (WQP) physical and chemical monitoring data from 2021-2023. The project studies water-quality patterns and exploratory low dissolved oxygen (DO) risk using structured/tabular environmental data.

The project is designed for a data mining final project. It applies a complete workflow: data acquisition, preprocessing, exploratory data analysis, correlation analysis, PCA, clustering, supervised classification, model evaluation, visualization, and comparison with related water-quality machine-learning work.

## Dataset

Expected raw Git LFS ZIP files:

- `data/resultphyschem (2).zip`
- `data/station (1).zip`

The EDA notebook extracts these files if needed:

- `data/resultphyschem.csv`
- `data/station.csv`

Verified current baseline:

- Result rows: 3,148,368
- Station rows: 19,948
- Date range: 2021-01-01 through 2023-12-31
- Years present: 2021, 2022, 2023
- Unique characteristics: 168
- Duplicate `ResultIdentifier` rows: 0

The earlier 2014-only dataset was incorrect and is not current project evidence.

## Methods

Main project methods:

- data cleaning and preprocessing
- exploratory data analysis
- Spearman correlation analysis
- principal component analysis (PCA)
- k-means clustering
- logistic regression
- decision tree
- random forest
- gradient boosting comparison
- low-DO threshold sensitivity
- probability-threshold analysis
- permutation importance

The supervised task uses a project-defined low-DO screening target: site-year-season median dissolved oxygen <5 mg/L. This is not a regulatory assessment, pollution prediction, causal inference, future forecast, or long-term trend model.

## Main Results

The final supervised presentation model is **random forest**.

Random forest grouped held-out test metrics:

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

Gradient boosting was tested as a literature-aligned boosted-tree comparison. It improved held-out precision but reduced recall, F1, ROC-AUC, PR-AUC, and grouped-CV F1 compared with random forest, so it remains a comparison model rather than the final headline model.

## Notebook-Only Reproduction

The recommended clean-checkout workflow is notebook-only and does not require the Python scripts.

Run from the repository root in Windows PowerShell:

```powershell
git lfs pull
jupyter notebook
```

Then run the notebooks in order:

1. `notebooks/florida_wqp_eda_workflow.ipynb`
2. `notebooks/florida_wqp_modeling_workflow.ipynb`

The EDA notebook writes:

- `data/processed/`
- `outputs/eda/`
- `outputs/figures/`

The modeling notebook reads those canonical EDA outputs and writes:

- `outputs/modeling/`
- `outputs/figures/modeling/`

The notebooks do not import from or call the scripts. The scripts are preserved as optional/reference command-line workflows.

## Key Reports

- Main report draft: `reports/final_ieee_project_report.md`
- Submission guide: `reports/submission_package_recommendation.md`
- Notebook reproducibility fix report: `reports/notebook_only_reproducibility_fix_report.md`
- Modeling results: `reports/wqp_2021_2023_modeling_results.md`
- Supervised modeling report: `reports/wqp_2021_2023_supervised_modeling_report.md`
- Literature comparison: `reports/wqp_2021_2023_modeling_literature_comparison.md`
- Presentation plan: `reports/wqp_2021_2023_final_presentation_plan.md`

## Expected Outputs

Important output folders:

- `outputs/eda/`
- `outputs/modeling/`
- `outputs/figures/`
- `outputs/figures/modeling/`

Important modeling outputs:

- `outputs/modeling/wqp_2021_2023_low_do_supervised_metrics.csv`
- `outputs/modeling/wqp_2021_2023_low_do_supervised_cv_metrics.csv`
- `outputs/modeling/wqp_2021_2023_low_do_supervised_confusion_matrices.csv`
- `outputs/modeling/wqp_2021_2023_low_do_literature_aligned_model_comparison.csv`
- `outputs/modeling/wqp_2021_2023_low_do_threshold_sensitivity_metrics.csv`
- `outputs/modeling/wqp_2021_2023_low_do_rf_probability_threshold_analysis.csv`
- `outputs/modeling/wqp_2021_2023_low_do_permutation_importance.csv`

Important figures:

- `outputs/figures/top_20_characteristics.png`
- `outputs/figures/records_by_year.png`
- `outputs/figures/modeling/wqp_2021_2023_core_pca_by_site_type.png`
- `outputs/figures/modeling/wqp_2021_2023_cluster_profile_heatmap.png`
- `outputs/figures/modeling/wqp_2021_2023_low_do_model_metric_comparison.png`
- `outputs/figures/modeling/wqp_2021_2023_low_do_confusion_matrix.png`
- `outputs/figures/modeling/wqp_2021_2023_low_do_feature_importance.png`

## Git LFS Note

The raw ZIP files should be tracked through Git LFS if they are submitted through GitHub:

- `data/resultphyschem (2).zip`
- `data/station (1).zip`

Do not commit extracted raw CSV files directly. See:

- `reports/git_lfs_data_submission_guide.md`
- `reports/submission_package_recommendation.md`


## Limitations

- WQP data are observational and unevenly sampled by site, provider, site type, season, and characteristic.
- Three years support exploratory seasonal and limited inter-year summaries, not long-term trend claims.
- The low-DO threshold is project-defined.
- Feature importance describes model behavior, not environmental causality.
- Model metrics should not be compared one-to-one with published WQI or DO forecasting studies that use different targets, splits, and datasets.

