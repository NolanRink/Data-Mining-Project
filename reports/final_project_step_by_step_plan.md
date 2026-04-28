# Final Water Quality Data Mining Project Step-by-Step Plan

## Current repo structure

Existing and recommended locations:

- `annotated-Project_Proposal.docx.pdf`: approved project proposal.
- `final_project_requirements.txt`: final project and proposal requirements.
- `resultphyschem.zip`: raw downloaded archive; preserve as received.
- `resultphyschem/resultphyschem.csv`: raw extracted Water Quality Portal physical/chemical results.
- `notebooks/illinois_water_quality_project_plan.ipynb`: lightweight planning and dataset exploration notebook.
- `reports/final_project_step_by_step_plan.md`: this step-by-step plan.
- `outputs/`: write derived cleaned tables, figures, and final exported visuals here.

Recommended final structure:

- `notebooks/illinois_water_quality_project_plan.ipynb`
- `notebooks/illinois_water_quality_final_analysis.ipynb`
- `reports/final_project_step_by_step_plan.md`
- `reports/final_report.md` or final report document
- `outputs/figures/` for exported plots
- `outputs/tables/` for cleaned summaries

Avoid adding more folders unless a final report or presentation export needs them.

## Proposal and requirements summary

Project goal: analyze spatial and seasonal patterns in Illinois water quality indicators from the public Water Quality Portal export, focusing on dissolved oxygen, pH, phosphorus, and specific conductance. The file spans 2021-2025, but final claims should emphasize seasonal comparisons and site-level patterns because usable volume is concentrated in 2021-2022. Dissolved oxygen saturation can be used as a supporting indicator.

Required deliverables from `final_project_requirements.txt`:

- Written report, 5-6 pages.
- Code submission for preprocessing, analysis/modeling, and visualization.
- Approximately 15 minute presentation.

Success criteria:

- Use public, structured/tabular environmental data.
- Keep data size medium to large but runnable on a normal laptop or desktop.
- Do not use image datasets or GPU-based methods.
- Apply multiple data mining or machine learning techniques.
- Interpret and visualize results.
- Compare results with existing or previously reported findings.

Proposal findings to preserve:

- Dataset source: Water Quality Portal, Illinois-focused physical/chemical water quality export.
- Observed size: 85,619 rows and 81 columns.
- Observed scope: 857 monitoring sites and 42 characteristics.
- Parsed date coverage: 2021-01-03 through 2025-12-31.
- Related work/comparison sources named in the proposal: Read et al. (2017) on the Water Quality Portal, Mohebbi and Akbariyeh (2024) on WQP nitrogen trends, Illinois EPA 2024 Integrated Water Quality Report, Illinois EPA river and stream monitoring information, and Illinois water quality standards in 35 Ill. Adm. Code Part 302.

## A. Project objective

Refined research question:

How do dissolved oxygen, pH, phosphorus, and specific conductance vary across Illinois monitoring sites and seasons, and can site-level indicator profiles reveal meaningful groups of monitoring locations?

Why it matters:

These indicators are core measures of stream and river condition. Dissolved oxygen and pH are directly relevant to aquatic life conditions, phosphorus is connected to nutrient enrichment, and specific conductance can indicate dissolved ions, runoff, or other watershed influences. Studying spatial and seasonal patterns can show whether water quality differs mainly by time of year, by monitoring location, or by indicator-specific behavior.

Final analysis should answer:

- Which indicators have the strongest seasonal patterns?
- Which monitoring sites have consistently high or low indicator values?
- Are there geographic patterns in the selected indicators across Illinois?
- Do site-level profiles cluster into interpretable groups?
- Are the observed results consistent with existing Water Quality Portal literature, Illinois EPA monitoring context, and relevant Illinois water quality standards?

## B. Dataset understanding

Source:

- Water Quality Portal public export, Illinois physical/chemical water quality results.

Observed dataset size:

- 85,619 rows.
- 81 columns.
- Approximately 53 MB CSV; about 243 MB in pandas memory during profiling.

Key fields observed:

- Site fields: `MonitoringLocationIdentifier`, `MonitoringLocationName`.
- Coordinates: `ActivityLocation/LatitudeMeasure`, `ActivityLocation/LongitudeMeasure`.
- Time fields: `ActivityStartDate`, `ActivityStartTime/Time`, `ActivityStartTime/TimeZoneCode`.
- Measurement fields: `CharacteristicName`, `ResultMeasureValue`, `ResultMeasure/MeasureUnitCode`, `ResultSampleFractionText`.
- Quality/context fields: `ResultDetectionConditionText`, `MeasureQualifierCode`, `ResultStatusIdentifier`, `DetectionQuantitationLimitTypeName`, `DetectionQuantitationLimitMeasure/MeasureValue`.
- Identifiers/audit fields: `ActivityIdentifier`, `ResultIdentifier`, `LastUpdated`, `ProviderName`.

Observed coverage:

- Date range: 2021-01-03 to 2025-12-31.
- Rows by year: 2021 has 38,137 rows; 2022 has 45,208 rows; 2023 has 852 rows; 2024 has 803 rows; 2025 has 619 rows.
- Monitoring sites: 857 unique `MonitoringLocationIdentifier` values.
- Site names: 405 unique `MonitoringLocationName` values, so names are reused and site ID should be the main key.
- Characteristics: 42 unique `CharacteristicName` values.
- Numeric result values: 77,099 parse as numeric; 8,520 are missing or non-numeric.

Primary indicators observed:

- `Dissolved oxygen (DO)`: 8,738 rows, 8,733 numeric, 381 sites, mostly `mg/L`.
- `pH`: 8,827 rows, 8,822 numeric, 402 sites, unit code missing as expected for pH.
- `Phosphorus`: 6,529 rows, 6,346 numeric, 673 sites, mostly `mg/L`, split across `Total` and `Dissolved` fractions. These fractions must be modeled as separate features, not combined into one phosphorus median.
- `Specific conductance`: 8,688 rows, 8,681 numeric, 353 sites, `umho/cm` and `uS/cm`.
- `Dissolved oxygen saturation`: 8,581 rows, 8,576 numeric, 352 sites, `%`.

Important data caveats:

- Later years have far fewer rows, so avoid trend claims across 2021-2025. Treat year-level summaries as coverage-aware context only.
- Many metadata columns are nearly or completely missing.
- Several indicators have outliers or physically questionable values that should be flagged before interpretation.
- Duplicate measurement keys exist depending on the key definition; do not delete rows until duplicate cause is understood.

## C. Preprocessing plan

1. Preserve raw data.

   Keep `resultphyschem/resultphyschem.csv` unchanged. Write all cleaned or summarized outputs under `outputs/`.

2. Select working columns.

   Use only stable columns needed for the analysis: site ID/name, latitude, longitude, activity date/time, characteristic, value, unit, fraction, status, detection/qualifier fields, activity ID, result ID, and project/source fields.

3. Parse dates and time variables.

   Convert `ActivityStartDate` to `activity_date`. Create `year`, `month`, and `season` where seasons are Winter, Spring, Summer, and Fall.

4. Parse numeric result values.

   Convert `ResultMeasureValue` to `result_value_num` with `pd.to_numeric(errors="coerce")`. Keep the raw text value for audit.

5. Filter target indicators.

   Start with exact names: `Dissolved oxygen (DO)`, `pH`, `Phosphorus`, `Specific conductance`, and `Dissolved oxygen saturation`. Avoid broad substring filters for pH because they also match phosphorus-related names.

6. Reconcile units.

   - Treat pH as unitless.
   - Keep dissolved oxygen in `mg/L`.
   - Keep phosphorus in `mg/L`, and create separate features such as `Phosphorus_Total` and `Phosphorus_Dissolved` from `ResultSampleFractionText`.
   - Treat `umho/cm` and `uS/cm` for specific conductance as equivalent for this project.
   - Keep dissolved oxygen saturation in `%`.

7. Handle duplicates conservatively.

   Use `ResultIdentifier` for row identity. For analysis keys, inspect repeated rows by site/date/time/characteristic/fraction/unit. If repeated values are exact or near-exact, summarize with median for site-season tables. If repeats differ by method/fraction/unit, keep them separate.

8. Handle missing values.

   Drop missing numeric values only for analyses requiring numeric input. Summarize missingness by characteristic and quality fields in the report. Do not impute raw water quality measurements for the main analysis.

9. Flag outliers.

   Create outlier/data-quality flags using indicator-specific rules and IQR checks. Examples from profiling:

   - Dissolved oxygen has max 746 mg/L and 55 IQR outlier rows.
   - pH includes 0 and max 12.35, with 55 IQR outlier rows.
   - Phosphorus has max 14.7 mg/L and 542 IQR outlier rows.
   - Specific conductance has max 4,980 and 271 IQR outlier rows.
   - Dissolved oxygen saturation reaches 300% and has 172 IQR outlier rows.

   Use flags for sensitivity checks. Do not silently delete outliers.

10. Create summary tables.

   Build:

   - Row-level cleaned primary-indicator table.
   - Seasonal indicator summary by characteristic/year/season.
   - Site-level median/mean table by site and clean indicator feature.
   - Wide site profile table for Spearman correlation and clustering, with phosphorus fractions kept separate.

## D. Exploratory analysis plan

Overall summaries:

- Count rows, sites, years, characteristics, and numeric coverage.
- Summarize populated versus sparse columns.
- Summarize quality/status fields.

Indicator distributions:

- Histograms or density plots for each primary indicator.
- Boxplots for each indicator, with outliers either shown or separately flagged.
- Separate phosphorus by `ResultSampleFractionText` when comparing total versus dissolved phosphorus.

Seasonal patterns:

- Boxplots by season for each indicator.
- Median seasonal summary table.
- Limited year-context summaries, with a clear caveat that 2023-2025 row counts are much smaller.

Site-level comparisons:

- Rank sites by median value for each indicator, requiring a minimum number of observations.
- Compare high- and low-median site groups.
- Show number of observations per site so sparse sites are not overinterpreted.

Spatial plots:

- Longitude/latitude scatterplots colored by site-level median indicator values.
- Facet or separate maps for each primary indicator.
- Include a note that these are coordinate scatterplots, not full GIS boundary maps.

Missingness and quality:

- Bar chart of missing values by key column.
- Table of non-numeric values and detection condition counts by characteristic.
- Duplicate key summary table.

## E. Data mining / ML methods plan

Method 1: Spearman correlation between indicators.

- Why it fits: shows monotonic relationships among water quality indicators at site or site-season level without assuming normal distributions or linear relationships.
- Input table: wide site or site-season profile with median indicator values, using clean features such as `Phosphorus_Total` and `Phosphorus_Dissolved`.
- Expected output: correlation matrix and heatmap.
- Interpretation: positive/negative associations, such as whether high conductance sites also show different phosphorus or oxygen profiles.
- Risks: correlations depend on aggregation choice and can be distorted by sparse sites or outliers.

Method 2: Seasonal comparison.

- Why it fits: directly addresses the proposal's seasonal pattern question.
- Input table: cleaned primary-indicator row table with `season` and `year`.
- Expected output: seasonal medians, boxplots, and simple statistical comparisons if appropriate.
- Interpretation: identify indicators with higher/lower typical values by season.
- Risks: seasons and years are not evenly sampled; later-year coverage is thin.

Method 3: Clustering monitoring sites.

- Why it fits: groups monitoring sites by multivariate water quality profile.
- Input table: site-level wide profile with median values for dissolved oxygen, pH, specific conductance, dissolved oxygen saturation, `Phosphorus_Total`, and `Phosphorus_Dissolved`; require a minimum observation count per feature per site.
- Expected output: cluster labels, PCA visualization, and table of cluster centers.
- Interpretation: describe clusters as high-conductance, high-phosphorus, lower-oxygen, or typical-background groups if the centers support that.
- Required preprocessing: use complete cases for the selected features, require at least 3-5 observations per feature per site, and scale all numeric features before clustering.
- Risks: clustering can create groups even when patterns are weak; scaling, sparse site coverage, phosphorus fraction handling, and missing-data handling strongly affect results.

Method 4: Outlier/data-quality flagging.

- Why it fits: water quality data includes extreme values that may represent true events, data entry errors, or unusual conditions.
- Input table: cleaned numeric primary-indicator records or site-season summaries.
- Expected output: flagged unusual measurements or unusual site-season combinations based on IQR and simple domain checks.
- Interpretation: discuss whether anomalies align with high phosphorus, very high conductance, low oxygen, or questionable data quality.
- Risks: outlier flagging can overfocus on data errors; keep it secondary and use it to qualify interpretation.

Recommended final methods:

- Primary: seasonal comparison and site-level clustering.
- Supporting: Spearman correlation and outlier/data-quality flagging.
- De-emphasized: classification, regression, and heavy anomaly detection unless the instructor explicitly asks for supervised modeling.

## F. Visualization plan

Recommended visuals for final report and presentation:

- Dataset coverage table: rows, columns, sites, characteristics, date range.
- Rows by year bar chart to show uneven temporal coverage.
- Missingness bar chart for key columns.
- Indicator distribution histograms or boxplots.
- Seasonal boxplots for each primary indicator.
- Limited year-context median summaries, with row counts shown or discussed.
- Latitude/longitude scatterplots colored by site-level median indicator values.
- Spearman correlation heatmap for site-level indicator profiles.
- PCA scatterplot of scaled site-level features colored by cluster label.
- Cluster summary table showing median indicator values per cluster.
- Small table of outlier flags and duplicate-key findings.

## Comparison with existing results

Use the existing-results comparison as a benchmark section, not as a formal regulatory compliance assessment. The WQP export is not guaranteed to contain the sampling frequency needed to determine legal attainment.

Concrete comparison targets:

- pH: compare observed pH distributions with the Illinois general-use pH range of 6.5 to 9.0 in 35 Ill. Adm. Code 302.204, while noting that the standard allows exceptions due to natural causes.
- Dissolved oxygen: compare low dissolved oxygen observations with the seasonal Illinois dissolved oxygen context in 35 Ill. Adm. Code 302.206. The standard includes different minimum and averaged values by season, so the final report should discuss low-DO patterns qualitatively unless the data support the required daily-minimum/daily-mean calculations.
- Phosphorus: compare total phosphorus summaries with the Illinois phosphorus context in 35 Ill. Adm. Code 302.205, which is specific to qualifying reservoirs/lakes and streams entering those waters. Do not apply that threshold blindly to every stream site.
- Nutrients and monitoring context: compare phosphorus patterns with Illinois EPA nutrient/eutrophication concerns and the Illinois EPA 2024 Integrated Water Quality Report.
- WQP data context: compare the project workflow and limitations with Read et al. (2017), which describes WQP as a large integrated water-quality data source that requires careful harmonization.

## G. 5-6 page report outline

1. Problem definition, about 0.5 page.

   State the refined research question, why Illinois water quality indicators matter, and what spatial/seasonal patterns the project investigates.

2. Dataset description, about 0.75 page.

   Describe the Water Quality Portal source, Illinois scope, 85,619 rows, 81 columns, 857 sites, 42 characteristics, 2021-2025 coverage, and primary fields.

3. Preprocessing, about 1 page.

   Explain column selection, date parsing, numeric parsing, unit handling, pH as unitless, conductance unit reconciliation, phosphorus fraction features, duplicate checks, missingness, and outlier/data-quality flags.

4. Methods, about 1 page.

   Describe EDA, seasonal comparison, Spearman correlation, clustering, and outlier/data-quality flagging. Include why each method fits the research question.

5. Results and visualizations, about 1.5-2 pages.

   Present the strongest distribution, seasonal, spatial, correlation, and clustering findings. Keep figures readable and connect each result to the research question.

6. Comparison with existing findings, about 0.5 page.

   Compare with Water Quality Portal literature, Illinois EPA monitoring context, and relevant Illinois standards for pH, dissolved oxygen, and phosphorus. Avoid claiming legal compliance unless the required sampling design supports it.

7. Conclusion, about 0.25-0.5 page.

   Summarize main findings, limitations, and practical next steps.

## H. 15 minute presentation outline

Slide 1: Title and research question, 1 minute.

- Main point: Illinois water quality patterns across sites and seasons.
- Visual: concise title with map/scatter background if available.

Slide 2: Dataset and source, 1.5 minutes.

- Main point: public WQP tabular data, 85,619 rows, 857 sites, 42 characteristics.
- Visual: dataset coverage table.

Slide 3: Key indicators, 1 minute.

- Main point: dissolved oxygen, pH, phosphorus, specific conductance, oxygen saturation.
- Visual: indicator summary table with rows/sites/units.

Slide 4: Data preparation, 1.5 minutes.

- Main point: date parsing, numeric parsing, unit checks, duplicate checks, outlier flags.
- Visual: preprocessing workflow diagram or bullet table.

Slide 5: Data coverage and quality, 1.5 minutes.

- Main point: coverage is strong in 2021-2022 but thinner in 2023-2025.
- Visual: rows by year chart and missingness/duplicate summary.

Slide 6: Indicator distributions, 1.5 minutes.

- Main point: typical ranges and extreme values.
- Visual: boxplots or histograms.

Slide 7: Seasonal patterns, 2 minutes.

- Main point: indicators vary by season if supported by analysis.
- Visual: seasonal boxplots or coverage-aware year-context medians.

Slide 8: Spatial/site patterns, 2 minutes.

- Main point: sites differ geographically and by median indicator values.
- Visual: latitude/longitude scatterplots.

Slide 9: Data mining results, 2 minutes.

- Main point: Spearman correlation and clustering reveal interpretable site profiles if supported.
- Visual: Spearman correlation heatmap and PCA/cluster plot.

Slide 10: Comparison and conclusion, 1 minute.

- Main point: connect results to WQP/Illinois EPA monitoring context, limitations, and final takeaway.
- Visual: short conclusion table.

## I. Milestone checklist

Milestone 1: data cleaning complete.

- Finalize primary indicator filter.
- Split phosphorus into `Phosphorus_Total` and `Phosphorus_Dissolved`.
- Parse date/year/month/season.
- Parse numeric values.
- Reconcile units.
- Flag outliers/data-quality issues and duplicates.
- Save cleaned primary-indicator table to `outputs/tables/`.

Milestone 2: EDA complete.

- Produce coverage, missingness, distribution, seasonal, and site-level summaries.
- Save report-ready figures to `outputs/figures/`.

Milestone 3: modeling/data mining complete.

- Build site-level wide table.
- Run Spearman correlation analysis.
- Run clustering with scaled features and minimum site observation thresholds.
- Create PCA or other 2D cluster visualization.

Milestone 4: results interpreted.

- Identify 3-5 defensible findings.
- Separate strong findings from limitations.
- Compare findings with proposal-related sources and concrete Illinois pH, dissolved oxygen, and phosphorus benchmarks where applicable.

Milestone 5: report drafted.

- Fill 5-6 page report sections.
- Insert final visuals and tables.
- Add limitations about temporal imbalance, sparse metadata, phosphorus fraction handling, and outlier/data-quality flags.

Milestone 6: presentation drafted.

- Build 10 slide deck.
- Use the same visuals as the report.
- Practice to approximately 15 minutes.

## Assumptions and risks

Assumptions:

- The raw CSV is the intended final dataset.
- The project can use pandas, numpy, matplotlib, and scikit-learn without installing new packages.
- The final analysis should prioritize reproducibility and interpretability over complex modeling.

Risks:

- 2023-2025 rows are sparse compared with 2021-2022, so year summaries are context only.
- Site names are not unique, so site ID must be used.
- pH unit codes are missing, but this is expected because pH is unitless.
- Conductance uses two equivalent-looking unit codes that should be documented.
- Phosphorus total and dissolved fractions must not be combined in modeling features.
- Outliers may be true events or data issues; flag them and report sensitivity rather than deleting them silently.

## Validation commands

From the project root:

```powershell
& 'C:\Users\nolan\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -c "import json, pathlib; json.load(open('notebooks/illinois_water_quality_project_plan.ipynb', encoding='utf-8')); print('notebook_json_ok')"
& 'C:\Users\nolan\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -c "import pandas as pd; df = pd.read_csv('resultphyschem/resultphyschem.csv', nrows=5, low_memory=False); print(df.shape); print(df.columns[:8].tolist())"
```

To open the notebook, use Jupyter if available:

```powershell
jupyter lab notebooks/illinois_water_quality_project_plan.ipynb
```

If `jupyter lab` is unavailable, try:

```powershell
jupyter notebook notebooks/illinois_water_quality_project_plan.ipynb
```
