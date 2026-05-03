"""
First-phase modeling for the 2021-2023 Florida WQP project.

This script starts from the fresh EDA and processed outputs. It creates
modeling audits, site-level Spearman correlations, PCA outputs, cautious
k-means clustering diagnostics, supervised target feasibility tables,
temporal readiness checks, and leakage-controlled low-DO supervised models.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text


BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
EDA_DIR = BASE_DIR / "outputs" / "eda"
REVIEW_DIR = BASE_DIR / "outputs" / "review"
MODELING_DIR = BASE_DIR / "outputs" / "modeling"
MODELING_FIG_DIR = BASE_DIR / "outputs" / "figures" / "modeling"

SITE_MATRIX_PATH = PROCESSED_DIR / "florida_wqp_site_characteristic_matrix.csv"
NUMERIC_LONG_PATH = PROCESSED_DIR / "florida_wqp_numeric_long.csv"
SITE_SUMMARY_PATH = EDA_DIR / "wqp_site_summary.csv"
CANDIDATE_PATH = EDA_DIR / "wqp_candidate_modeling_variables.csv"
MATRIX_VALIDATION_PATH = EDA_DIR / "wqp_candidate_matrix_validation.csv"
RANGE_SCREENING_PATH = EDA_DIR / "wqp_numeric_range_screening.csv"
PROCESSED_VALIDATION_PATH = EDA_DIR / "wqp_processed_validation_summary.csv"
CANDIDATE_REVIEW_PATH = REVIEW_DIR / "wqp_2021_2023_candidate_variable_review.csv"

EXPECTED_YEARS = {2021, 2022, 2023}
FIG_DPI = 220
RANDOM_STATE = 42
K_RANGE = range(2, 9)

CORE_VARIABLES = ["pH", "Temperature, water", "Dissolved oxygen (DO)"]
EXPANDED_VARIABLES = CORE_VARIABLES + [
    "Dissolved oxygen saturation",
    "Phosphorus",
    "Ammonia",
    "Orthophosphate",
]
AUDIT_VARIABLES = EXPANDED_VARIABLES + [
    "Specific conductance",
    "Salinity",
    "Turbidity",
    "Escherichia coli",
    "Nitrate + Nitrite",
    "Inorganic nitrogen (nitrate and nitrite) ***retired***use Nitrate + Nitrite",
]

MIN_CORE_MEASURED = 2
MIN_EXPANDED_MEASURED = 4
MIN_EXPANDED_SITES = 1000

# Conservative impossibility filters for site-level medians. Nutrient high
# values are audited but not upper-capped in this first pass.
FEATURE_LIMITS = {
    "pH": (0.0, 14.0),
    "Temperature, water": (0.0, 40.0),
    "Dissolved oxygen (DO)": (0.0, 20.0),
    "Dissolved oxygen saturation": (0.0, 200.0),
    "Phosphorus": (0.0, None),
    "Ammonia": (0.0, None),
    "Orthophosphate": (0.0, None),
}

TARGET_SPECS = [
    {
        "target": "low_dissolved_oxygen_lt_5_mg_L",
        "characteristic": "Dissolved oxygen (DO)",
        "unit": "mg/L",
        "threshold": 5.0,
        "direction": "lt",
        "definition": "result value < 5 mg/L",
        "exclude_predictors": {"Dissolved oxygen (DO)", "Dissolved oxygen saturation"},
    },
    {
        "target": "high_phosphorus_ge_0_1_mg_L",
        "characteristic": "Phosphorus",
        "unit": "mg/L",
        "threshold": 0.1,
        "direction": "ge",
        "definition": "result value >= 0.1 mg/L",
        "exclude_predictors": {"Phosphorus", "Orthophosphate"},
    },
    {
        "target": "high_ammonia_ge_0_1_mg_L",
        "characteristic": "Ammonia",
        "unit": "mg/L",
        "threshold": 0.1,
        "direction": "ge",
        "definition": "result value >= 0.1 mg/L",
        "exclude_predictors": {"Ammonia"},
    },
    {
        "target": "high_orthophosphate_ge_0_1_mg_L",
        "characteristic": "Orthophosphate",
        "unit": "mg/L",
        "threshold": 0.1,
        "direction": "ge",
        "definition": "result value >= 0.1 mg/L",
        "exclude_predictors": {"Orthophosphate", "Phosphorus"},
    },
    {
        "target": "high_turbidity_ge_10_NTU",
        "characteristic": "Turbidity",
        "unit": "NTU",
        "threshold": 10.0,
        "direction": "ge",
        "definition": "result value >= 10 NTU",
        "exclude_predictors": {"Turbidity"},
    },
    {
        "target": "e_coli_ge_235_MPN_100mL",
        "characteristic": "Escherichia coli",
        "unit": "MPN/100mL",
        "threshold": 235.0,
        "direction": "ge",
        "definition": "result value >= 235 MPN/100mL",
        "exclude_predictors": {"Escherichia coli"},
    },
]

TEMPORAL_VARIABLES = EXPANDED_VARIABLES + [
    "Turbidity",
    "Nitrate + Nitrite",
    "Escherichia coli",
    "Specific conductance",
    "Salinity",
]

LOW_DO_THRESHOLDS = [5.0, 4.0, 3.0]
LOW_DO_TARGET = "Dissolved oxygen (DO)"
LOW_DO_LEAKAGE_PREDICTORS = {"Dissolved oxygen (DO)", "Dissolved oxygen saturation"}
LOW_DO_PREDICTOR_UNITS = {
    "pH": "Unitless",
    "Temperature, water": "deg C",
    "Phosphorus": "mg/L",
    "Ammonia": "mg/L",
    "Orthophosphate": "mg/L",
    "Turbidity": "NTU",
}
LOW_DO_ALL_VARIABLE_UNITS = {
    LOW_DO_TARGET: "mg/L",
    "Dissolved oxygen saturation": "%",
    **LOW_DO_PREDICTOR_UNITS,
}
LOW_DO_EXCLUDED_CANDIDATES = {
    "Specific conductance": "unit harmonization unresolved",
    "Salinity": "unit harmonization unresolved",
}
LOW_DO_DESIGNS = [
    {
        "design_level": "site",
        "group_cols": ["MonitoringLocationIdentifier"],
        "min_predictor_count": 3,
        "preferred": False,
    },
    {
        "design_level": "site_year",
        "group_cols": ["MonitoringLocationIdentifier", "year"],
        "min_predictor_count": 3,
        "preferred": False,
    },
    {
        "design_level": "site_year_season",
        "group_cols": ["MonitoringLocationIdentifier", "year", "season"],
        "min_predictor_count": 3,
        "preferred": True,
    },
]
LOW_DO_TRAINING_DESIGN = "site_year_season"
LOW_DO_TRAINING_TARGET_COL = "low_do_lt_5_0_mg_L"
LOW_DO_MIN_PREDICTORS = 3
LOW_DO_TEST_SIZE = 0.2
LOW_DO_CV_SPLITS = 5
LOW_DO_RF_PROBABILITY_THRESHOLDS = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
LOW_DO_PERMUTATION_REPEATS = 5


def ensure_dirs() -> None:
    MODELING_DIR.mkdir(parents=True, exist_ok=True)
    MODELING_FIG_DIR.mkdir(parents=True, exist_ok=True)


def save_current_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()


def load_inputs() -> dict[str, pd.DataFrame]:
    print("Loading current EDA and processed inputs")
    return {
        "site_matrix": pd.read_csv(SITE_MATRIX_PATH),
        "site_summary": pd.read_csv(SITE_SUMMARY_PATH),
        "candidates": pd.read_csv(CANDIDATE_PATH),
        "matrix_validation": pd.read_csv(MATRIX_VALIDATION_PATH),
        "ranges": pd.read_csv(RANGE_SCREENING_PATH),
        "processed_validation": pd.read_csv(PROCESSED_VALIDATION_PATH),
        "candidate_review": pd.read_csv(CANDIDATE_REVIEW_PATH),
    }


def validation_value(processed_validation: pd.DataFrame, check: str) -> str:
    row = processed_validation.loc[processed_validation["check"].eq(check), "value"]
    return "" if row.empty else str(row.iloc[0])


def write_modeling_input_audit(inputs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    print("Writing modeling input audit")
    site_matrix = inputs["site_matrix"]
    candidates = inputs["candidates"]
    matrix_validation = inputs["matrix_validation"]
    processed_validation = inputs["processed_validation"]
    site_matrix_vars = set(site_matrix.columns) - {"MonitoringLocationIdentifier"}
    candidate_vars = set(candidates["CharacteristicName"])

    min_year = validation_value(processed_validation, "active_year_min")
    max_year = validation_value(processed_validation, "active_year_max")
    expected_years = validation_value(processed_validation, "expected_2021_2023_years_present")
    unexpected_years = validation_value(processed_validation, "unexpected_activity_years")

    rows = [
        {
            "check": "site_matrix_exists",
            "status": SITE_MATRIX_PATH.exists(),
            "value": str(SITE_MATRIX_PATH),
            "notes": "Processed site-characteristic matrix input.",
        },
        {
            "check": "site_matrix_shape",
            "status": True,
            "value": f"{site_matrix.shape[0]} rows x {site_matrix.shape[1]} columns",
            "notes": "Current EDA expected 18,339 rows x 21 columns; document if changed after future EDA reruns.",
        },
        {
            "check": "candidate_metadata_exists",
            "status": CANDIDATE_PATH.exists(),
            "value": str(CANDIDATE_PATH),
            "notes": "Fresh candidate metadata.",
        },
        {
            "check": "matrix_validation_exists",
            "status": MATRIX_VALIDATION_PATH.exists(),
            "value": str(MATRIX_VALIDATION_PATH),
            "notes": "Fresh matrix validation output.",
        },
        {
            "check": "range_screening_exists",
            "status": RANGE_SCREENING_PATH.exists(),
            "value": str(RANGE_SCREENING_PATH),
            "notes": "Fresh numeric range screening output.",
        },
        {
            "check": "matrix_variables_match_candidate_metadata",
            "status": site_matrix_vars.issubset(candidate_vars),
            "value": f"{len(site_matrix_vars)} matrix variables; {len(candidate_vars)} candidate variables",
            "notes": "All matrix variables should appear in current candidate metadata.",
        },
        {
            "check": "core_variables_present",
            "status": all(var in site_matrix.columns for var in CORE_VARIABLES),
            "value": "; ".join(f"{var}={var in site_matrix.columns}" for var in CORE_VARIABLES),
            "notes": "Required first-pass variables.",
        },
        {
            "check": "date_coverage_2021_2023",
            "status": min_year == "2021" and max_year == "2023" and expected_years == "True",
            "value": f"min_year={min_year}; max_year={max_year}; expected_years={expected_years}; unexpected={unexpected_years}",
            "notes": "Modeling must use the corrected 2021-2023 baseline.",
        },
        {
            "check": "pH_in_site_matrix",
            "status": "pH" in site_matrix.columns,
            "value": str("pH" in site_matrix.columns),
            "notes": "pH should remain available for first-pass modeling.",
        },
        {
            "check": "input_no_current_2014_only_claim",
            "status": min_year != "2014" and max_year != "2014",
            "value": f"{min_year}-{max_year}",
            "notes": "No active modeling input should describe the current export as 2014-only.",
        },
        {
            "check": "new_modeling_output_directory",
            "status": MODELING_DIR.exists(),
            "value": str(MODELING_DIR),
            "notes": "Created by this run for fresh 2021-2023 outputs.",
        },
        {
            "check": "new_modeling_figure_directory",
            "status": MODELING_FIG_DIR.exists(),
            "value": str(MODELING_FIG_DIR),
            "notes": "Created by this run for fresh 2021-2023 figures.",
        },
        {
            "check": "old_modeling_outputs_reused",
            "status": True,
            "value": "False",
            "notes": "This replacement script writes new 2021-2023 outputs and does not read old modeling artifacts.",
        },
    ]
    audit = pd.DataFrame(rows)
    audit.to_csv(MODELING_DIR / "wqp_2021_2023_modeling_input_audit.csv", index=False)
    if not audit["status"].astype(bool).all():
        failed = audit.loc[~audit["status"].astype(bool), "check"].tolist()
        raise RuntimeError(f"Modeling input audit failed: {failed}")
    return audit


def audit_selected_variables(inputs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    print("Writing selected-variable audit")
    site_matrix = inputs["site_matrix"]
    candidates = inputs["candidates"].set_index("CharacteristicName", drop=False)
    matrix_validation = inputs["matrix_validation"].set_index("CharacteristicName", drop=False)
    ranges = inputs["ranges"].set_index("CharacteristicName", drop=False)

    rows = []
    for variable in AUDIT_VARIABLES:
        candidate = candidates.loc[variable] if variable in candidates.index else None
        matrix_row = matrix_validation.loc[variable] if variable in matrix_validation.index else None
        range_row = ranges.loc[variable] if variable in ranges.index else None
        present_in_matrix = variable in site_matrix.columns
        missing_pct = float(site_matrix[variable].isna().mean() * 100) if present_in_matrix else np.nan
        include_core = variable in CORE_VARIABLES and present_in_matrix and missing_pct <= 55
        include_expanded = (
            variable in EXPANDED_VARIABLES
            and present_in_matrix
            and missing_pct <= 80
            and "retired" not in variable.lower()
        )
        defer = not include_core and not include_expanded

        reason_parts = []
        if variable in {"Specific conductance", "Salinity"}:
            reason_parts.append("defer until unit harmonization is resolved")
            defer = True
            include_expanded = False
        if variable == "Escherichia coli":
            reason_parts.append("defer from PCA/clustering because matrix missingness is high")
            defer = True
            include_expanded = False
        if "retired" in variable.lower():
            reason_parts.append("exclude retired duplicate nitrogen label")
            defer = True
            include_expanded = False
        if present_in_matrix and pd.notna(missing_pct):
            reason_parts.append(f"matrix missingness {missing_pct:.1f}%")
        else:
            reason_parts.append("not present in current site matrix")
        if range_row is not None:
            reason_parts.append(
                f"range flags: negatives={int(range_row['negative_count'])}, "
                f"IQR-high={int(range_row['extreme_high_count_iqr'])}, max={range_row['max']:.4g}"
            )

        rows.append(
            {
                "variable": variable,
                "present_in_candidate_metadata": candidate is not None,
                "present_in_site_matrix": present_in_matrix,
                "dominant_unit": "" if candidate is None else candidate.get("dominant_unit", ""),
                "site_count": np.nan if matrix_row is None else matrix_row.get("site_count", np.nan),
                "active_years": np.nan if candidate is None else candidate.get("active_years", np.nan),
                "matrix_missingness_pct": missing_pct,
                "range_outlier_concerns": "; ".join(reason_parts),
                "include_in_core_model": include_core,
                "include_in_expanded_model": include_expanded,
                "defer": defer,
                "reason": "; ".join(reason_parts),
            }
        )

    audit = pd.DataFrame(rows)
    audit.to_csv(MODELING_DIR / "wqp_2021_2023_selected_variable_audit.csv", index=False)
    return audit


def apply_range_filters(values: pd.DataFrame, variables: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    filtered = values.copy()
    rows = []
    for variable in variables:
        series = pd.to_numeric(filtered[variable], errors="coerce")
        original_missing = int(series.isna().sum())
        low, high = FEATURE_LIMITS.get(variable, (None, None))
        low_mask = pd.Series(False, index=series.index)
        high_mask = pd.Series(False, index=series.index)
        if low is not None:
            low_mask = series < low
            series = series.mask(low_mask)
        if high is not None:
            high_mask = series > high
            series = series.mask(high_mask)
        filtered[variable] = series
        rows.append(
            {
                "variable": variable,
                "missing_before_range_filter": original_missing,
                "values_below_allowed_range": int(low_mask.sum()),
                "values_above_allowed_range": int(high_mask.sum()),
                "missing_after_range_filter": int(series.isna().sum()),
                "allowed_min": low,
                "allowed_max": high,
            }
        )
    return filtered, pd.DataFrame(rows)


def build_feature_matrix(
    label: str,
    site_matrix: pd.DataFrame,
    site_summary: pd.DataFrame,
    variables: list[str],
    min_measured: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print(f"Building {label} feature matrix")
    metadata_cols = [
        "MonitoringLocationIdentifier",
        "site_type",
        "county_code",
        "huc8",
        "latitude",
        "longitude",
        "record_count",
        "active_years",
        "active_months",
        "modeling_sparse_flag",
    ]
    metadata = site_summary[[col for col in metadata_cols if col in site_summary.columns]].copy()
    raw_values = site_matrix[["MonitoringLocationIdentifier"] + variables].copy()
    raw_values[variables] = raw_values[variables].apply(pd.to_numeric, errors="coerce")
    filtered_values, range_summary = apply_range_filters(raw_values[variables], variables)
    measured_count = filtered_values.notna().sum(axis=1)
    eligible_mask = measured_count >= min_measured
    eligible_values = filtered_values.loc[eligible_mask].copy()
    eligible_ids = raw_values.loc[eligible_mask, ["MonitoringLocationIdentifier"]].copy()

    imputer = SimpleImputer(strategy="median")
    imputed_array = imputer.fit_transform(eligible_values)
    imputed = pd.DataFrame(imputed_array, columns=variables, index=eligible_values.index)

    scaler = StandardScaler()
    standardized = pd.DataFrame(
        scaler.fit_transform(imputed),
        columns=variables,
        index=eligible_values.index,
    )

    feature_matrix = pd.concat([eligible_ids.reset_index(drop=True), imputed.reset_index(drop=True)], axis=1)
    feature_matrix = feature_matrix.merge(metadata, on="MonitoringLocationIdentifier", how="left")
    feature_matrix["selected_variable_count"] = len(variables)
    feature_matrix["measured_selected_variable_count"] = measured_count.loc[eligible_mask].to_numpy()
    feature_matrix["imputed_selected_value_count"] = len(variables) - feature_matrix["measured_selected_variable_count"]
    feature_matrix["matrix_version"] = label

    missing_rows = []
    for variable, median_value in zip(variables, imputer.statistics_):
        before_missing = int(filtered_values[variable].isna().sum())
        eligible_missing = int(eligible_values[variable].isna().sum())
        missing_rows.append(
            {
                "matrix_version": label,
                "variable": variable,
                "total_sites_before_site_filter": len(filtered_values),
                "eligible_sites_after_site_filter": len(eligible_values),
                "missing_before_site_filter": before_missing,
                "missing_pct_before_site_filter": before_missing / len(filtered_values) * 100,
                "missing_in_eligible_sites_before_imputation": eligible_missing,
                "missing_pct_in_eligible_sites_before_imputation": eligible_missing / len(eligible_values) * 100,
                "imputed_value": median_value,
                "missing_after_imputation": int(imputed[variable].isna().sum()),
                "minimum_measured_variables_required": min_measured,
            }
        )
    missing_summary = pd.DataFrame(missing_rows)

    output_path = MODELING_DIR / f"wqp_2021_2023_{label}_feature_matrix.csv"
    feature_matrix.to_csv(output_path, index=False)
    range_summary.insert(0, "matrix_version", label)
    return feature_matrix, eligible_values, imputed, standardized, pd.concat([missing_summary, range_summary], ignore_index=True, sort=False)


def write_spearman_outputs(label: str, values: pd.DataFrame, variables: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    print(f"Running {label} Spearman correlation")
    corr = values[variables].corr(method="spearman")
    pairwise_counts = values[variables].notna().astype(int).T.dot(values[variables].notna().astype(int))
    corr.to_csv(MODELING_DIR / f"wqp_2021_2023_{label}_spearman_correlations.csv")
    pairwise_counts.to_csv(MODELING_DIR / f"wqp_2021_2023_{label}_spearman_pairwise_counts.csv")

    plt.figure(figsize=(max(6, len(variables) * 0.9), max(5, len(variables) * 0.75)))
    sns.heatmap(corr, cmap="vlag", center=0, vmin=-1, vmax=1, annot=True, fmt=".2f", square=True)
    plt.title(f"2021-2023 {label.title()} Site-Level Spearman Correlation")
    save_current_figure(MODELING_FIG_DIR / f"wqp_2021_2023_{label}_correlation_heatmap.png")
    return corr, pairwise_counts


def run_pca(
    label: str,
    feature_matrix: pd.DataFrame,
    standardized: pd.DataFrame,
    variables: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print(f"Running {label} PCA")
    n_components = min(len(variables), 5)
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    scores_array = pca.fit_transform(standardized[variables])
    pc_cols = [f"PC{i + 1}" for i in range(n_components)]

    scores = pd.DataFrame(scores_array, columns=pc_cols)
    score_metadata_cols = [
        "MonitoringLocationIdentifier",
        "site_type",
        "county_code",
        "huc8",
        "record_count",
        "active_years",
        "active_months",
        "modeling_sparse_flag",
        "measured_selected_variable_count",
        "imputed_selected_value_count",
        "matrix_version",
    ]
    scores = pd.concat(
        [feature_matrix[[col for col in score_metadata_cols if col in feature_matrix.columns]].reset_index(drop=True), scores],
        axis=1,
    )

    loadings = pd.DataFrame(pca.components_.T, columns=pc_cols)
    loadings.insert(0, "variable", variables)
    explained = pd.DataFrame(
        {
            "component": pc_cols,
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_explained_variance_ratio": np.cumsum(pca.explained_variance_ratio_),
        }
    )

    scores.to_csv(MODELING_DIR / f"wqp_2021_2023_{label}_pca_scores.csv", index=False)
    loadings.to_csv(MODELING_DIR / f"wqp_2021_2023_{label}_pca_loadings.csv", index=False)
    explained.to_csv(MODELING_DIR / f"wqp_2021_2023_{label}_pca_explained_variance.csv", index=False)

    if n_components >= 2:
        plt.figure(figsize=(9, 6))
        sns.scatterplot(
            data=scores,
            x="PC1",
            y="PC2",
            hue="site_type" if "site_type" in scores.columns else None,
            s=18,
            alpha=0.55,
            linewidth=0,
        )
        plt.title(f"2021-2023 {label.title()} PCA by Site Type")
        if "site_type" in scores.columns:
            plt.legend(title="Site type", bbox_to_anchor=(1.02, 1), loc="upper left")
        save_current_figure(MODELING_FIG_DIR / f"wqp_2021_2023_{label}_pca_by_site_type.png")

    plot_cols = [col for col in ["PC1", "PC2", "PC3"] if col in loadings.columns]
    plot_data = loadings.set_index("variable")[plot_cols]
    plt.figure(figsize=(max(7, len(variables) * 0.8), 4.8))
    sns.heatmap(plot_data, cmap="vlag", center=0, annot=True, fmt=".2f")
    plt.title(f"2021-2023 {label.title()} PCA Loadings")
    save_current_figure(MODELING_FIG_DIR / f"wqp_2021_2023_{label}_pca_loadings.png")
    return scores, loadings, explained


def choose_cluster_solution(scores: pd.DataFrame) -> tuple[int, str]:
    stable = scores[scores["min_cluster_pct"] >= 5].copy()
    if not stable.empty:
        chosen = stable.sort_values(["silhouette_score", "min_cluster_pct"], ascending=[False, False]).iloc[0]
        return int(chosen["k"]), "Highest silhouette among k values with every cluster at least 5% of sites."
    chosen = scores.sort_values(["min_cluster_pct", "silhouette_score"], ascending=[False, False]).iloc[0]
    return int(chosen["k"]), "No k met the 5% minimum cluster-size guardrail; selected most balanced diagnostic solution."


def run_clustering(
    label: str,
    feature_matrix: pd.DataFrame,
    standardized: pd.DataFrame,
    pca_scores: pd.DataFrame,
    variables: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, int, str]:
    print(f"Running quality-controlled clustering on {label} matrix")
    score_rows = []
    labels_by_k: dict[int, np.ndarray] = {}
    for k in K_RANGE:
        model = KMeans(n_clusters=k, n_init=25, random_state=RANDOM_STATE)
        labels = model.fit_predict(standardized[variables])
        labels_by_k[k] = labels
        sizes = pd.Series(labels).value_counts().sort_index()
        score_rows.append(
            {
                "matrix_version": label,
                "k": k,
                "silhouette_score": float(silhouette_score(standardized[variables], labels)),
                "min_cluster_size": int(sizes.min()),
                "max_cluster_size": int(sizes.max()),
                "min_cluster_pct": float(sizes.min() / len(labels) * 100),
                "max_cluster_pct": float(sizes.max() / len(labels) * 100),
                "small_cluster_flag": bool((sizes / len(labels) * 100 < 5).any()),
                "cluster_size_distribution": "; ".join(f"{idx}:{count}" for idx, count in sizes.items()),
            }
        )
    cluster_scores = pd.DataFrame(score_rows)
    chosen_k, chosen_reason = choose_cluster_solution(cluster_scores)
    cluster_scores["chosen"] = cluster_scores["k"].eq(chosen_k)
    cluster_scores["chosen_reason"] = np.where(cluster_scores["chosen"], chosen_reason, "")
    cluster_scores.to_csv(MODELING_DIR / "wqp_2021_2023_cluster_scores.csv", index=False)

    assignments = feature_matrix[
        [
            "MonitoringLocationIdentifier",
            "site_type",
            "county_code",
            "huc8",
            "record_count",
            "active_years",
            "active_months",
            "modeling_sparse_flag",
            "measured_selected_variable_count",
            "imputed_selected_value_count",
        ]
    ].copy()
    assignments["cluster"] = labels_by_k[chosen_k]
    if {"PC1", "PC2"}.issubset(pca_scores.columns):
        assignments = assignments.merge(
            pca_scores[["MonitoringLocationIdentifier", "PC1", "PC2"]],
            on="MonitoringLocationIdentifier",
            how="left",
        )
    assignments.to_csv(MODELING_DIR / "wqp_2021_2023_site_clusters.csv", index=False)

    profile_rows = []
    raw_values = feature_matrix[["MonitoringLocationIdentifier"] + variables].merge(
        assignments[["MonitoringLocationIdentifier", "cluster"]],
        on="MonitoringLocationIdentifier",
        how="left",
    )
    for cluster, group in raw_values.groupby("cluster"):
        row = {"cluster": cluster, "site_count": len(group)}
        for variable in variables:
            row[f"{variable}_median"] = group[variable].median()
            row[f"{variable}_mean"] = group[variable].mean()
        profile_rows.append(row)
    cluster_profiles = pd.DataFrame(profile_rows).sort_values("cluster")
    cluster_profiles.to_csv(MODELING_DIR / "wqp_2021_2023_cluster_profiles.csv", index=False)

    metadata_rows = []
    for cluster, group in assignments.groupby("cluster"):
        site_type_counts = group["site_type"].fillna("Unknown").value_counts().head(5)
        metadata_rows.append(
            {
                "cluster": cluster,
                "site_count": len(group),
                "top_site_types": "; ".join(f"{idx}: {count}" for idx, count in site_type_counts.items()),
                "median_record_count": float(group["record_count"].median()),
                "median_active_months": float(group["active_months"].median()),
                "modeling_sparse_site_count": int(group["modeling_sparse_flag"].fillna(False).sum()),
                "median_imputed_selected_values": float(group["imputed_selected_value_count"].median()),
            }
        )
    metadata_summary = pd.DataFrame(metadata_rows).sort_values("cluster")
    metadata_summary.to_csv(MODELING_DIR / "wqp_2021_2023_cluster_metadata_summary.csv", index=False)

    plt.figure(figsize=(8, 5))
    sns.lineplot(data=cluster_scores, x="k", y="silhouette_score", marker="o")
    for _, row in cluster_scores.iterrows():
        if row["small_cluster_flag"]:
            plt.scatter(row["k"], row["silhouette_score"], color="red", s=45, zorder=3)
    plt.axvline(chosen_k, color="black", linestyle="--", linewidth=1, label=f"chosen k={chosen_k}")
    plt.title("2021-2023 K-Means Diagnostic Silhouette Scores")
    plt.xlabel("k")
    plt.ylabel("Silhouette score")
    plt.legend()
    save_current_figure(MODELING_FIG_DIR / "wqp_2021_2023_cluster_silhouette_scores.png")

    if {"PC1", "PC2"}.issubset(assignments.columns):
        plt.figure(figsize=(8.5, 6))
        sns.scatterplot(data=assignments, x="PC1", y="PC2", hue="cluster", palette="tab10", s=20, alpha=0.65, linewidth=0)
        plt.title("2021-2023 PCA Scores by K-Means Cluster")
        plt.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
        save_current_figure(MODELING_FIG_DIR / "wqp_2021_2023_pca_by_cluster.png")

    median_cols = [col for col in cluster_profiles.columns if col.endswith("_median")]
    heatmap_data = cluster_profiles.set_index("cluster")[median_cols]
    heatmap_data.columns = [col.replace("_median", "") for col in heatmap_data.columns]
    scaled = (heatmap_data - heatmap_data.mean()) / heatmap_data.std(ddof=0).replace(0, np.nan)
    plt.figure(figsize=(max(7, len(variables) * 1.1), 4.8))
    sns.heatmap(scaled, cmap="vlag", center=0, annot=heatmap_data.round(2), fmt="", cbar_kws={"label": "cluster median z-score"})
    plt.title("2021-2023 Cluster Profile Medians")
    plt.ylabel("Cluster")
    save_current_figure(MODELING_FIG_DIR / "wqp_2021_2023_cluster_profile_heatmap.png")
    return cluster_scores, assignments, cluster_profiles, metadata_summary, chosen_k, chosen_reason


def target_positive_mask(values: pd.Series, direction: str, threshold: float) -> pd.Series:
    if direction == "lt":
        return values < threshold
    return values >= threshold


def low_do_target_column(threshold: float) -> str:
    return f"low_do_lt_{str(threshold).replace('.', '_')}_mg_L"


def scan_numeric_long() -> tuple[pd.DataFrame, pd.DataFrame]:
    print("Scanning processed long data for target feasibility and temporal readiness")
    target_counts = {
        spec["target"]: {
            "target": spec["target"],
            "characteristic": spec["characteristic"],
            "threshold": spec["threshold"],
            "threshold_source": "project-defined",
            "target_definition": spec["definition"],
            "sample_count": 0,
            "positive_count": 0,
            "site_ids": set(),
            "positive_site_ids": set(),
        }
        for spec in TARGET_SPECS
    }
    temporal_years: dict[tuple[str, str], set[int]] = defaultdict(set)
    temporal_seasons: dict[tuple[str, str], set[str]] = defaultdict(set)
    temporal_counts: dict[tuple[str, str], int] = defaultdict(int)

    target_by_char = defaultdict(list)
    for spec in TARGET_SPECS:
        target_by_char[spec["characteristic"]].append(spec)

    usecols = [
        "MonitoringLocationIdentifier",
        "CharacteristicName",
        "normalized_result_unit",
        "result_value_numeric",
        "year",
        "season",
    ]
    scan_vars = set(TEMPORAL_VARIABLES) | set(target_by_char)
    for chunk in pd.read_csv(NUMERIC_LONG_PATH, usecols=usecols, chunksize=250_000, low_memory=False):
        chunk = chunk[chunk["CharacteristicName"].isin(scan_vars)].copy()
        if chunk.empty:
            continue
        chunk["result_value_numeric"] = pd.to_numeric(chunk["result_value_numeric"], errors="coerce")
        chunk = chunk[chunk["result_value_numeric"].notna()]
        if chunk.empty:
            continue

        for spec in TARGET_SPECS:
            data = chunk[
                chunk["CharacteristicName"].eq(spec["characteristic"])
                & chunk["normalized_result_unit"].eq(spec["unit"])
            ]
            if data.empty:
                continue
            mask = target_positive_mask(data["result_value_numeric"], spec["direction"], spec["threshold"])
            counts = target_counts[spec["target"]]
            counts["sample_count"] += int(len(data))
            counts["positive_count"] += int(mask.sum())
            counts["site_ids"].update(data["MonitoringLocationIdentifier"].dropna().astype(str).unique())
            counts["positive_site_ids"].update(data.loc[mask, "MonitoringLocationIdentifier"].dropna().astype(str).unique())

        for _, row in chunk[["MonitoringLocationIdentifier", "CharacteristicName", "year", "season"]].dropna().iterrows():
            key = (str(row["CharacteristicName"]), str(row["MonitoringLocationIdentifier"]))
            try:
                temporal_years[key].add(int(row["year"]))
            except (TypeError, ValueError):
                pass
            temporal_seasons[key].add(str(row["season"]))
            temporal_counts[key] += 1

    target_rows = []
    for spec in TARGET_SPECS:
        counts = target_counts[spec["target"]]
        sample_count = counts["sample_count"]
        positive_count = counts["positive_count"]
        negative_count = sample_count - positive_count
        positive_rate = positive_count / sample_count * 100 if sample_count else 0.0
        site_count = len(counts["site_ids"])
        positive_site_count = len(counts["positive_site_ids"])
        predictors = [v for v in EXPANDED_VARIABLES if v not in spec["exclude_predictors"]]
        feasible = sample_count >= 20_000 and 5 <= positive_rate <= 95 and site_count >= 1000
        missingness_note = "Use site/time predictor co-occurrence checks before training."
        if spec["characteristic"] == "Escherichia coli":
            missingness_note = "Sparse in site matrix; feasible only for a narrower bacteria-focused supervised table."
        if spec["target"] == "high_turbidity_ge_10_NTU":
            missingness_note = "Skewed target with high outlier burden; use caution."
        target_rows.append(
            {
                "target": spec["target"],
                "target_definition": spec["definition"],
                "threshold": spec["threshold"],
                "threshold_source": "project-defined",
                "sample_count": sample_count,
                "positive_count": positive_count,
                "negative_count": negative_count,
                "positive_rate_pct": positive_rate,
                "site_count": site_count,
                "positive_site_count": positive_site_count,
                "predictor_variables_after_leakage_exclusion": "; ".join(predictors),
                "leakage_concerns": f"Exclude direct target variables: {', '.join(sorted(spec['exclude_predictors']))}.",
                "missingness_concerns": missingness_note,
                "feasible": feasible,
                "recommendation": "screen further before training; no supervised model trained in this pass" if feasible else "defer; target feasibility is weak or incomplete",
            }
        )
    target_feasibility = pd.DataFrame(target_rows)
    target_feasibility.to_csv(MODELING_DIR / "wqp_2021_2023_supervised_target_feasibility.csv", index=False)

    temporal_rows = []
    for variable in TEMPORAL_VARIABLES:
        keys = [key for key in temporal_years if key[0] == variable]
        site_count = len(keys)
        all_three = sum(EXPECTED_YEARS.issubset(temporal_years[key]) for key in keys)
        seasonal = sum(len(temporal_seasons[key]) >= 4 for key in keys)
        total_records = sum(temporal_counts[key] for key in keys)
        year_feasible = all_three >= 1000
        season_feasible = seasonal >= 1000
        notes = []
        if variable in {"Specific conductance", "Salinity"}:
            notes.append("unit harmonization unresolved")
        if variable == "Escherichia coli":
            notes.append("sparse compared with chemistry variables")
        if not year_feasible:
            notes.append("limited all-three-year site continuity")
        temporal_rows.append(
            {
                "variable": variable,
                "record_count_in_processed_long": total_records,
                "site_count_with_numeric_records": site_count,
                "site_count_with_all_three_years_present": all_three,
                "site_count_with_all_four_seasons_present": seasonal,
                "year_comparison_feasible": year_feasible,
                "seasonal_comparison_feasible": season_feasible,
                "trend_language_should_be_avoided": True,
                "notes": "; ".join(notes) if notes else "Use as limited seasonal/inter-year comparison after continuity filtering.",
            }
        )
    temporal_readiness = pd.DataFrame(temporal_rows)
    temporal_readiness.to_csv(MODELING_DIR / "wqp_2021_2023_temporal_modeling_readiness.csv", index=False)
    return target_feasibility, temporal_readiness


def season_sort_value(value: object) -> int:
    order = {"Winter": 0, "Spring": 1, "Summer": 2, "Fall": 3}
    return order.get(str(value), 99)


def first_non_null(values: pd.Series) -> object:
    values = values.dropna()
    return values.iloc[0] if not values.empty else np.nan


def load_low_do_rows() -> pd.DataFrame:
    print("Loading processed long data for low-DO supervised readiness checks")
    usecols = [
        "MonitoringLocationIdentifier",
        "CharacteristicName",
        "normalized_result_unit",
        "result_value_numeric",
        "year",
        "month",
        "season",
        "site_type",
    ]
    chunks = []
    needed = set(LOW_DO_ALL_VARIABLE_UNITS)
    for chunk in pd.read_csv(NUMERIC_LONG_PATH, usecols=usecols, chunksize=250_000, low_memory=False):
        chunk = chunk[chunk["CharacteristicName"].isin(needed)].copy()
        if chunk.empty:
            continue
        chunk["result_value_numeric"] = pd.to_numeric(chunk["result_value_numeric"], errors="coerce")
        chunk = chunk[chunk["result_value_numeric"].notna()]
        if chunk.empty:
            continue
        expected_units = chunk["CharacteristicName"].map(LOW_DO_ALL_VARIABLE_UNITS)
        chunk = chunk[chunk["normalized_result_unit"].eq(expected_units)]
        if not chunk.empty:
            chunks.append(chunk)
    if not chunks:
        raise RuntimeError("No processed long rows were available for low-DO readiness checks.")
    return pd.concat(chunks, ignore_index=True)


def build_low_do_design_tables(long_rows: pd.DataFrame) -> dict[str, pd.DataFrame]:
    print("Building low-DO supervised design table candidates")
    design_tables = {}
    for design in LOW_DO_DESIGNS:
        design_level = design["design_level"]
        group_cols = design["group_cols"]
        values = (
            long_rows.groupby(group_cols + ["CharacteristicName"], dropna=False)["result_value_numeric"]
            .median()
            .unstack("CharacteristicName")
            .reset_index()
        )
        metadata_aggs = {"site_type": ("site_type", first_non_null)}
        if "month" not in group_cols and "month" in long_rows.columns:
            metadata_aggs["month_count"] = ("month", "nunique")
        if "season" not in group_cols and "season" in long_rows.columns:
            metadata_aggs["season_count"] = ("season", "nunique")
        metadata = long_rows.groupby(group_cols, dropna=False).agg(**metadata_aggs).reset_index()
        table = values.merge(metadata, on=group_cols, how="left")
        if "season" in table.columns:
            table["season_sort"] = table["season"].map(season_sort_value)
        for threshold in LOW_DO_THRESHOLDS:
            target_col = low_do_target_column(threshold)
            target_values = pd.Series(pd.NA, index=table.index, dtype="boolean")
            target_ready = table[LOW_DO_TARGET].notna()
            target_values.loc[target_ready] = table.loc[target_ready, LOW_DO_TARGET] < threshold
            table[target_col] = target_values
        predictor_cols = [col for col in LOW_DO_PREDICTOR_UNITS if col in table.columns]
        table["available_low_do_predictor_count"] = table[predictor_cols].notna().sum(axis=1)
        table["low_do_target_ready"] = table[LOW_DO_TARGET].notna()
        table["design_level"] = design_level
        table.to_csv(MODELING_DIR / f"wqp_2021_2023_low_do_{design_level}_design_table.csv", index=False)
        design_tables[design_level] = table
    return design_tables


def class_balance_usable(positive_rate: float, positives: int, negatives: int) -> bool:
    return positives >= 100 and negatives >= 100 and 5 <= positive_rate <= 95


def review_low_do_design_tables(design_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    predictor_cols = list(LOW_DO_PREDICTOR_UNITS)
    for design in LOW_DO_DESIGNS:
        design_level = design["design_level"]
        table = design_tables[design_level]
        target_ready = table[table["low_do_target_ready"]].copy()
        default_target = target_ready["low_do_lt_5_0_mg_L"].astype(bool) if not target_ready.empty else pd.Series(dtype=bool)
        available_predictors = [col for col in predictor_cols if col in table.columns]
        min_predictors = design["min_predictor_count"]
        rows_retained = int((target_ready[available_predictors].notna().sum(axis=1) >= min_predictors).sum()) if available_predictors else 0
        year_coverage = ""
        if "year" in target_ready.columns and not target_ready.empty:
            year_coverage = "; ".join(str(int(year)) for year in sorted(target_ready["year"].dropna().unique()))
        season_coverage = ""
        if "season" in target_ready.columns and not target_ready.empty:
            seasons = sorted(target_ready["season"].dropna().unique(), key=season_sort_value)
            season_coverage = "; ".join(str(season) for season in seasons)
        positives = int(default_target.sum()) if len(default_target) else 0
        negatives = int(len(default_target) - positives)
        positive_rate = positives / len(default_target) * 100 if len(default_target) else 0.0
        feasible = (
            len(target_ready) >= 1000
            and class_balance_usable(positive_rate, positives, negatives)
            and rows_retained >= 1000
            and len(available_predictors) >= 3
        )
        rows.append(
            {
                "design_level": design_level,
                "aggregation": "median by " + ", ".join(design["group_cols"]),
                "row_count": len(table),
                "unique_site_count": table["MonitoringLocationIdentifier"].nunique(),
                "year_coverage": year_coverage,
                "season_coverage": season_coverage,
                "target_ready_row_count": len(target_ready),
                "positive_count_lt_5": positives,
                "negative_count_lt_5": negatives,
                "positive_rate_pct_lt_5": positive_rate,
                "predictor_columns_available_after_leakage_exclusion": "; ".join(available_predictors),
                "missingness_by_predictor_pct": "; ".join(
                    f"{col}={target_ready[col].isna().mean() * 100:.1f}%" for col in available_predictors
                ),
                "rows_retained_after_predictor_missingness_filter": rows_retained,
                "minimum_predictor_count_required": min_predictors,
                "feasibility_recommendation": "feasible for next training pass" if feasible else "needs caution or additional filtering before training",
            }
        )
    design_review = pd.DataFrame(rows)
    design_review.to_csv(MODELING_DIR / "wqp_2021_2023_low_do_design_table_review.csv", index=False)
    return design_review


def review_low_do_thresholds(design_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for design_level, table in design_tables.items():
        target_ready = table[table["low_do_target_ready"]].copy()
        for threshold in LOW_DO_THRESHOLDS:
            target_col = low_do_target_column(threshold)
            positives = int(target_ready[target_col].eq(True).sum())
            total = len(target_ready)
            negatives = total - positives
            positive_rate = positives / total * 100 if total else 0.0
            usable = class_balance_usable(positive_rate, positives, negatives)
            rows.append(
                {
                    "design_level": design_level,
                    "threshold_value_mg_L": threshold,
                    "target_definition": f"median dissolved oxygen < {threshold:g} mg/L",
                    "threshold_source": "project-defined",
                    "total_target_ready_rows": total,
                    "positives": positives,
                    "negatives": negatives,
                    "positive_rate_pct": positive_rate,
                    "class_balance_usable": usable,
                    "recommendation": "usable screening threshold" if usable else "defer or use only as sensitivity threshold",
                }
            )
    threshold_review = pd.DataFrame(rows)
    threshold_review.to_csv(MODELING_DIR / "wqp_2021_2023_low_do_threshold_review.csv", index=False)
    return threshold_review


def review_low_do_predictors(design_tables: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    cooccurrence_rows = []
    missingness_rows = []
    predictors = list(LOW_DO_PREDICTOR_UNITS)
    for design in LOW_DO_DESIGNS:
        design_level = design["design_level"]
        table = design_tables[design_level]
        available_predictors = [col for col in predictors if col in table.columns]
        min_predictors = design["min_predictor_count"]
        target_ready = table[table["low_do_target_ready"]].copy()
        predictor_counts = target_ready[available_predictors].notna().sum(axis=1) if available_predictors else pd.Series(dtype=int)
        for threshold in LOW_DO_THRESHOLDS:
            target_col = low_do_target_column(threshold)
            target_mask = target_ready[target_col].eq(True)
            for predictor in predictors:
                present = predictor in table.columns
                predictor_nonmissing = target_ready[predictor].notna() if present else pd.Series(False, index=target_ready.index)
                cooccurrence_rows.append(
                    {
                        "design_level": design_level,
                        "threshold_value_mg_L": threshold,
                        "predictor": predictor,
                        "expected_unit": LOW_DO_PREDICTOR_UNITS[predictor],
                        "predictor_present": present,
                        "target_ready_rows": len(target_ready),
                        "target_predictor_cooccurrence_rows": int(predictor_nonmissing.sum()),
                        "target_predictor_cooccurrence_pct": float(predictor_nonmissing.mean() * 100) if len(target_ready) else 0.0,
                        "positive_rows_with_predictor": int((target_mask & predictor_nonmissing).sum()),
                        "negative_rows_with_predictor": int((~target_mask & predictor_nonmissing).sum()),
                        "included_after_leakage_exclusion": present,
                        "notes": "candidate non-leakage predictor" if present else "not available in this design table",
                    }
                )
            rows_all = int((predictor_counts == len(available_predictors)).sum()) if available_predictors else 0
            rows_min = int((predictor_counts >= min_predictors).sum()) if available_predictors else 0
            missingness_rows.append(
                {
                    "design_level": design_level,
                    "threshold_value_mg_L": threshold,
                    "candidate_predictor_count": len(available_predictors),
                    "minimum_predictor_count_required": min_predictors,
                    "target_ready_rows": len(target_ready),
                    "positive_count": int(target_mask.sum()),
                    "negative_count": int((~target_mask).sum()),
                    "positive_rate_pct": float(target_mask.mean() * 100) if len(target_ready) else 0.0,
                    "rows_with_all_predictors": rows_all,
                    "rows_with_all_predictors_pct": rows_all / len(target_ready) * 100 if len(target_ready) else 0.0,
                    "rows_with_minimum_predictor_count": rows_min,
                    "rows_with_minimum_predictor_count_pct": rows_min / len(target_ready) * 100 if len(target_ready) else 0.0,
                    "imputation_required": rows_all < len(target_ready),
                    "imputation_burden_acceptable": rows_min >= 1000 and rows_min / len(target_ready) >= 0.5 if len(target_ready) else False,
                    "metadata_domination_risk": "year/season/site_type should be used only with caution; chemistry predictors should remain primary",
                    "missingness_by_predictor_pct": "; ".join(
                        f"{col}={target_ready[col].isna().mean() * 100:.1f}%" for col in available_predictors
                    ),
                    "recommendation": "usable with missingness controls" if rows_min >= 1000 else "too sparse without more filtering",
                }
            )
    cooccurrence = pd.DataFrame(cooccurrence_rows)
    missingness = pd.DataFrame(missingness_rows)
    cooccurrence.to_csv(MODELING_DIR / "wqp_2021_2023_low_do_predictor_cooccurrence.csv", index=False)
    missingness.to_csv(MODELING_DIR / "wqp_2021_2023_low_do_missingness_readiness.csv", index=False)
    return cooccurrence, missingness


def review_low_do_split_feasibility(design_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for design in LOW_DO_DESIGNS:
        design_level = design["design_level"]
        table = design_tables[design_level]
        target_ready = table[table["low_do_target_ready"]].copy()
        for threshold in LOW_DO_THRESHOLDS:
            target_col = low_do_target_column(threshold)
            data = target_ready[["MonitoringLocationIdentifier", target_col] + ([ "year" ] if "year" in target_ready.columns else [])].dropna()
            if data.empty:
                rows.append(
                    {
                        "design_level": design_level,
                        "threshold_value_mg_L": threshold,
                        "unique_site_count": 0,
                        "positive_site_count": 0,
                        "negative_site_count": 0,
                        "mixed_site_count": 0,
                        "group_kfold_feasible": False,
                        "stratified_group_kfold_feasible": False,
                        "grouped_train_test_balance_feasible": False,
                        "year_holdout_supported": False,
                        "site_leakage_risk_if_random_row_split": "not evaluated",
                        "recommended_split_strategy": "not recommended",
                    }
                )
                continue
            site_flags = data.groupby("MonitoringLocationIdentifier")[target_col].agg(["min", "max", "count"])
            positive_sites = int(site_flags["max"].astype(bool).sum())
            negative_sites = int((~site_flags["min"].astype(bool)).sum())
            mixed_sites = int((site_flags["min"] != site_flags["max"]).sum())
            total = len(data)
            positives = int(data[target_col].astype(bool).sum())
            positive_rate = positives / total * 100 if total else 0.0
            year_holdout_supported = False
            if "year" in data.columns:
                year_counts = data.groupby("year")[target_col].agg(["count", "sum"])
                year_holdout_supported = bool(
                    len(year_counts) >= 3
                    and (year_counts["count"] - year_counts["sum"]).min() >= 100
                    and year_counts["sum"].min() >= 100
                )
            group_kfold_feasible = site_flags.shape[0] >= 5 and positive_sites >= 5 and negative_sites >= 5
            stratified_group_feasible = site_flags.shape[0] >= 20 and positive_sites >= 10 and negative_sites >= 10 and 5 <= positive_rate <= 95
            grouped_balance_feasible = site_flags.shape[0] >= 1000 and positive_sites >= 100 and negative_sites >= 100 and 5 <= positive_rate <= 95
            if design_level == "site":
                strategy = "site-level stratified random split"
                leakage_risk = "low; one row per site, but repeated-site leakage is not applicable"
            elif grouped_balance_feasible:
                strategy = "grouped-by-site train/test split; grouped cross-validation optional"
                leakage_risk = "high if rows are split randomly because sites repeat across rows"
            else:
                strategy = "not recommended until grouped class balance improves"
                leakage_risk = "high if rows are split randomly because sites repeat across rows"
            rows.append(
                {
                    "design_level": design_level,
                    "threshold_value_mg_L": threshold,
                    "row_count": total,
                    "positive_rate_pct": positive_rate,
                    "unique_site_count": int(site_flags.shape[0]),
                    "positive_site_count": positive_sites,
                    "negative_site_count": negative_sites,
                    "mixed_site_count": mixed_sites,
                    "group_kfold_feasible": group_kfold_feasible,
                    "stratified_group_kfold_feasible": stratified_group_feasible,
                    "grouped_train_test_balance_feasible": grouped_balance_feasible,
                    "year_holdout_supported": year_holdout_supported,
                    "site_leakage_risk_if_random_row_split": leakage_risk,
                    "recommended_split_strategy": strategy,
                }
            )
    split_feasibility = pd.DataFrame(rows)
    split_feasibility.to_csv(MODELING_DIR / "wqp_2021_2023_low_do_split_feasibility.csv", index=False)
    return split_feasibility


def write_low_do_leakage_guardrails() -> pd.DataFrame:
    rows = [
        {
            "variable_or_field": "Dissolved oxygen (DO)",
            "leakage_risk": "direct",
            "action": "exclude",
            "reason": "Directly defines the low-DO target.",
        },
        {
            "variable_or_field": "Dissolved oxygen saturation",
            "leakage_risk": "near_direct",
            "action": "exclude",
            "reason": "Closely derived from oxygen condition and can encode the target.",
        },
        {
            "variable_or_field": "ResultIdentifier",
            "leakage_risk": "direct_metadata",
            "action": "exclude",
            "reason": "Identifier field, not an environmental predictor.",
        },
        {
            "variable_or_field": "ActivityIdentifier",
            "leakage_risk": "direct_metadata",
            "action": "exclude",
            "reason": "Sampling event identifier, not a predictor.",
        },
        {
            "variable_or_field": "MonitoringLocationIdentifier",
            "leakage_risk": "direct_metadata",
            "action": "exclude",
            "reason": "Site ID can memorize site-specific target prevalence.",
        },
        {
            "variable_or_field": "raw result/date fields",
            "leakage_risk": "direct_metadata",
            "action": "exclude",
            "reason": "Use only intentional transformations such as year or season.",
        },
        {
            "variable_or_field": "target threshold fields",
            "leakage_risk": "direct",
            "action": "exclude",
            "reason": "Any derived target flag must not be a predictor.",
        },
        {
            "variable_or_field": "site_type",
            "leakage_risk": "low",
            "action": "allow_with_caution",
            "reason": "Can be useful metadata but may dominate or proxy sampling design.",
        },
        {
            "variable_or_field": "year",
            "leakage_risk": "low",
            "action": "allow_with_caution",
            "reason": "Allow only as intentional temporal context, not trend prediction.",
        },
        {
            "variable_or_field": "season",
            "leakage_risk": "low",
            "action": "allow_with_caution",
            "reason": "Season is interpretable but can dominate if sampling is uneven.",
        },
    ]
    for variable in LOW_DO_PREDICTOR_UNITS:
        rows.append(
            {
                "variable_or_field": variable,
                "leakage_risk": "low",
                "action": "allow",
                "reason": "Candidate non-leakage chemistry predictor if coverage supports it.",
            }
        )
    for variable, reason in LOW_DO_EXCLUDED_CANDIDATES.items():
        rows.append(
            {
                "variable_or_field": variable,
                "leakage_risk": "low",
                "action": "allow_with_caution",
                "reason": f"Not included in readiness predictors because {reason}.",
            }
        )
    guardrails = pd.DataFrame(rows)
    guardrails.to_csv(MODELING_DIR / "wqp_2021_2023_low_do_leakage_guardrails.csv", index=False)
    return guardrails


def choose_low_do_recommendation(
    threshold_review: pd.DataFrame,
    missingness: pd.DataFrame,
    split_feasibility: pd.DataFrame,
) -> pd.DataFrame:
    merged = threshold_review.merge(
        missingness,
        on=["design_level", "threshold_value_mg_L"],
        how="left",
        suffixes=("", "_missingness"),
    ).merge(
        split_feasibility,
        on=["design_level", "threshold_value_mg_L"],
        how="left",
        suffixes=("", "_split"),
    )
    merged["preferred_design"] = merged["design_level"].eq("site_year_season")
    merged["ready_score"] = (
        merged["class_balance_usable"].astype(int) * 3
        + merged["imputation_burden_acceptable"].astype(int) * 2
        + merged["grouped_train_test_balance_feasible"].fillna(False).astype(int) * 2
        + merged["year_holdout_supported"].fillna(False).astype(int)
        + merged["preferred_design"].astype(int)
    )
    candidates = merged[merged["threshold_value_mg_L"].eq(5.0)].copy()
    if candidates.empty:
        candidates = merged.copy()
    chosen = candidates.sort_values(
        ["ready_score", "preferred_design", "rows_with_minimum_predictor_count"],
        ascending=[False, False, False],
    ).iloc[0]
    recommended_predictors = list(LOW_DO_PREDICTOR_UNITS)
    blockers = []
    if chosen["threshold_source"] == "project-defined":
        blockers.append("threshold remains project-defined unless a domain source is added")
    if not bool(chosen["class_balance_usable"]):
        blockers.append("class balance is weak")
    if not bool(chosen["imputation_burden_acceptable"]):
        blockers.append("predictor missingness burden is high")
    if chosen["design_level"] != "site" and not bool(chosen["grouped_train_test_balance_feasible"]):
        blockers.append("grouped-by-site split balance is weak")
    ready = len([b for b in blockers if not b.startswith("threshold remains")]) == 0
    recommendation = pd.DataFrame(
        [
            {
                "target": "low_dissolved_oxygen",
                "recommended_design_level": chosen["design_level"],
                "recommended_threshold": f"median dissolved oxygen < {chosen['threshold_value_mg_L']:g} mg/L",
                "recommended_predictors": "; ".join(recommended_predictors),
                "excluded_leakage_predictors": "; ".join(sorted(LOW_DO_LEAKAGE_PREDICTORS)),
                "split_strategy": chosen["recommended_split_strategy"],
                "ready_for_training": "yes" if ready else "no",
                "blockers": "; ".join(blockers) if blockers else "none identified by readiness checks",
                "next_steps": "Use these readiness guardrails for training, reporting, and any later refinement; keep threshold language project-defined unless a source is added."
                if ready
                else "Address blockers before training.",
            }
        ]
    )
    recommendation.to_csv(MODELING_DIR / "wqp_2021_2023_supervised_readiness_recommendation.csv", index=False)
    return recommendation


def plot_low_do_readiness(threshold_review: pd.DataFrame, missingness: pd.DataFrame) -> None:
    plt.figure(figsize=(9, 5))
    plot_data = threshold_review.copy()
    plot_data["threshold_label"] = plot_data["threshold_value_mg_L"].map(lambda x: f"< {x:g} mg/L")
    sns.barplot(data=plot_data, x="design_level", y="positive_rate_pct", hue="threshold_label")
    plt.title("2021-2023 Low-DO Target Positive Rate by Design")
    plt.xlabel("Design level")
    plt.ylabel("Positive rate (%)")
    plt.legend(title="Threshold")
    save_current_figure(MODELING_FIG_DIR / "wqp_2021_2023_low_do_threshold_balance.png")

    selected = missingness[missingness["threshold_value_mg_L"].eq(5.0)].copy()
    rows = []
    for _, row in selected.iterrows():
        for item in str(row["missingness_by_predictor_pct"]).split("; "):
            if not item or "=" not in item:
                continue
            variable, pct = item.split("=")
            rows.append(
                {
                    "design_level": row["design_level"],
                    "predictor": variable,
                    "missingness_pct": float(pct.rstrip("%")),
                }
            )
    if rows:
        missing_plot = pd.DataFrame(rows)
        plt.figure(figsize=(10, 5.5))
        sns.barplot(data=missing_plot, x="predictor", y="missingness_pct", hue="design_level")
        plt.title("2021-2023 Low-DO Predictor Missingness at <5 mg/L Target")
        plt.xlabel("Predictor")
        plt.ylabel("Missingness among target-ready rows (%)")
        plt.xticks(rotation=35, ha="right")
        plt.legend(title="Design")
        save_current_figure(MODELING_FIG_DIR / "wqp_2021_2023_low_do_predictor_missingness.png")


def run_low_do_supervised_readiness() -> dict[str, pd.DataFrame]:
    print("Running low-DO supervised readiness checks without training models")
    long_rows = load_low_do_rows()
    design_tables = build_low_do_design_tables(long_rows)
    design_review = review_low_do_design_tables(design_tables)
    threshold_review = review_low_do_thresholds(design_tables)
    cooccurrence, missingness = review_low_do_predictors(design_tables)
    split_feasibility = review_low_do_split_feasibility(design_tables)
    leakage_guardrails = write_low_do_leakage_guardrails()
    recommendation = choose_low_do_recommendation(threshold_review, missingness, split_feasibility)
    plot_low_do_readiness(threshold_review, missingness)
    return {
        "design_tables": design_tables,
        "design_review": design_review,
        "threshold_review": threshold_review,
        "cooccurrence": cooccurrence,
        "missingness": missingness,
        "split_feasibility": split_feasibility,
        "leakage_guardrails": leakage_guardrails,
        "recommendation": recommendation,
    }


def safe_auc(y_true: pd.Series | np.ndarray, probabilities: np.ndarray, metric: str) -> float:
    if len(np.unique(y_true)) < 2:
        return np.nan
    if metric == "roc_auc":
        return float(roc_auc_score(y_true, probabilities))
    return float(average_precision_score(y_true, probabilities))


def evaluate_classifier(
    model_name: str,
    split_name: str,
    y_true: pd.Series | np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    row_count: int,
    site_count: int,
) -> dict[str, object]:
    return {
        "model": model_name,
        "split": split_name,
        "row_count": row_count,
        "unique_site_count": site_count,
        "positive_rate_pct": float(np.mean(y_true) * 100),
        "predicted_positive_rate_pct": float(np.mean(predictions) * 100),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": safe_auc(y_true, probabilities, "roc_auc"),
        "average_precision_pr_auc": safe_auc(y_true, probabilities, "average_precision"),
    }


def get_positive_probabilities(model: Pipeline, x_values: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x_values)[:, 1]
    return model.predict(x_values).astype(float)


def display_model_name(model_name: str) -> str:
    labels = {
        "majority_class_baseline": "Majority Baseline",
        "logistic_regression": "Logistic Regression",
        "decision_tree": "Decision Tree",
        "random_forest": "Random Forest",
        "gradient_boosting": "Gradient Boosting",
    }
    return labels.get(model_name, str(model_name).replace("_", " ").title())


def make_low_do_models() -> dict[str, Pipeline]:
    return {
        "majority_class_baseline": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", DummyClassifier(strategy="most_frequent")),
            ]
        ),
        "logistic_regression": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=1000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "decision_tree": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    DecisionTreeClassifier(
                        max_depth=5,
                        min_samples_leaf=100,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "gradient_boosting": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    GradientBoostingClassifier(
                        n_estimators=150,
                        learning_rate=0.05,
                        max_depth=3,
                        min_samples_leaf=50,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=150,
                        max_depth=8,
                        min_samples_leaf=50,
                        class_weight="balanced",
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def build_low_do_model_table_for_threshold(design_table: pd.DataFrame, threshold: float) -> pd.DataFrame:
    predictors = list(LOW_DO_PREDICTOR_UNITS)
    target_col = low_do_target_column(threshold)
    if target_col not in design_table.columns:
        raise ValueError(f"Missing low-DO target column: {target_col}")
    target_ready = design_table[design_table["low_do_target_ready"]].copy()
    model_table = target_ready[
        target_ready[predictors].notna().sum(axis=1) >= LOW_DO_MIN_PREDICTORS
    ].copy()
    model_table["target_low_do"] = model_table[target_col].astype(bool).astype(int)
    model_table["low_do_threshold_mg_L"] = threshold
    return model_table


def build_low_do_training_table(low_do_readiness: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("Building final low-DO supervised training table")
    design_table = low_do_readiness["design_tables"][LOW_DO_TRAINING_DESIGN].copy()
    predictors = list(LOW_DO_PREDICTOR_UNITS)
    excluded = sorted(
        LOW_DO_LEAKAGE_PREDICTORS
        | {
            "MonitoringLocationIdentifier",
            "ResultIdentifier",
            "ActivityIdentifier",
            "ActivityStartDate",
            "ActivityStartTime/Time",
            LOW_DO_TRAINING_TARGET_COL,
            "low_do_lt_4_0_mg_L",
            "low_do_lt_3_0_mg_L",
        }
    )
    target_ready = design_table[design_table["low_do_target_ready"]].copy()
    model_table = build_low_do_model_table_for_threshold(design_table, 5.0)

    missingness = "; ".join(f"{col}={model_table[col].isna().mean() * 100:.1f}%" for col in predictors)
    positives = int(model_table["target_low_do"].sum())
    total = len(model_table)
    summary = pd.DataFrame(
        [
            {
                "design_level": LOW_DO_TRAINING_DESIGN,
                "target": "low_dissolved_oxygen",
                "threshold": "median dissolved oxygen < 5 mg/L",
                "threshold_source": "project-defined",
                "total_rows_before_filter": len(design_table),
                "target_ready_rows_before_filter": len(target_ready),
                "rows_after_filter": total,
                "unique_sites": model_table["MonitoringLocationIdentifier"].nunique(),
                "positive_count": positives,
                "negative_count": total - positives,
                "positive_rate_pct": positives / total * 100 if total else 0.0,
                "predictors_used": "; ".join(predictors),
                "predictors_excluded": "; ".join(excluded),
                "missingness_by_predictor_after_filter": missingness,
                "minimum_predictor_count_required": LOW_DO_MIN_PREDICTORS,
                "split_strategy": "grouped-by-site train/test split",
                "notes": "Numeric predictors are imputed inside train/test and CV pipelines only.",
            }
        ]
    )
    summary.to_csv(MODELING_DIR / "wqp_2021_2023_low_do_training_table_summary.csv", index=False)
    return model_table, summary


def grouped_train_test_split(model_table: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=LOW_DO_TEST_SIZE, random_state=RANDOM_STATE)
    x_dummy = model_table[list(LOW_DO_PREDICTOR_UNITS)]
    y = model_table["target_low_do"]
    groups = model_table["MonitoringLocationIdentifier"]
    train_idx, test_idx = next(splitter.split(x_dummy, y, groups=groups))
    return train_idx, test_idx


def train_low_do_supervised_models(low_do_readiness: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame | str]:
    print("Training low-DO supervised models with grouped-by-site split")
    model_table, training_summary = build_low_do_training_table(low_do_readiness)
    predictors = list(LOW_DO_PREDICTOR_UNITS)
    train_idx, test_idx = grouped_train_test_split(model_table)

    train = model_table.iloc[train_idx].copy()
    test = model_table.iloc[test_idx].copy()
    x_train = train[predictors]
    y_train = train["target_low_do"]
    x_test = test[predictors]
    y_test = test["target_low_do"]
    train_sites = set(train["MonitoringLocationIdentifier"].astype(str))
    test_sites = set(test["MonitoringLocationIdentifier"].astype(str))
    overlap_sites = train_sites & test_sites

    models = make_low_do_models()
    metrics_rows = []
    confusion_rows = []
    fitted_models: dict[str, Pipeline] = {}

    for model_name, model in models.items():
        model.fit(x_train, y_train)
        fitted_models[model_name] = model
        for split_name, x_values, y_values, site_values in [
            ("train", x_train, y_train, train["MonitoringLocationIdentifier"]),
            ("test", x_test, y_test, test["MonitoringLocationIdentifier"]),
        ]:
            predictions = model.predict(x_values)
            probabilities = get_positive_probabilities(model, x_values)
            metrics_rows.append(
                evaluate_classifier(
                    model_name,
                    split_name,
                    y_values,
                    predictions,
                    probabilities,
                    len(y_values),
                    site_values.nunique(),
                )
            )
            cm = confusion_matrix(y_values, predictions, labels=[0, 1])
            confusion_rows.append(
                {
                    "model": model_name,
                    "split": split_name,
                    "true_negative": int(cm[0, 0]),
                    "false_positive": int(cm[0, 1]),
                    "false_negative": int(cm[1, 0]),
                    "true_positive": int(cm[1, 1]),
                }
            )

    metrics = pd.DataFrame(metrics_rows)
    metrics["train_row_count"] = len(train)
    metrics["test_row_count"] = len(test)
    metrics["unique_train_sites"] = len(train_sites)
    metrics["unique_test_sites"] = len(test_sites)
    metrics["train_test_site_overlap"] = len(overlap_sites)
    metrics.to_csv(MODELING_DIR / "wqp_2021_2023_low_do_supervised_metrics.csv", index=False)

    confusion = pd.DataFrame(confusion_rows)
    confusion.to_csv(MODELING_DIR / "wqp_2021_2023_low_do_supervised_confusion_matrices.csv", index=False)

    cv_metrics = run_low_do_grouped_cv(model_table, predictors, models)
    literature_model_comparison = write_literature_aligned_model_comparison(metrics, cv_metrics)
    write_low_do_interpretability_outputs(fitted_models, predictors)
    probability_thresholds = write_low_do_rf_probability_threshold_analysis(
        fitted_models["random_forest"],
        x_test,
        y_test,
    )
    rf_permutation_importance = write_low_do_permutation_importance(
        fitted_models["random_forest"],
        x_test,
        y_test,
        predictors,
    )
    threshold_sensitivity, threshold_sensitivity_cv = run_low_do_threshold_sensitivity(low_do_readiness)
    audit = write_low_do_supervised_audit(
        training_summary,
        predictors,
        train,
        test,
        overlap_sites,
    )
    plot_low_do_supervised_figures(metrics, cv_metrics, confusion)
    return {
        "training_summary": training_summary,
        "metrics": metrics,
        "cv_metrics": cv_metrics,
        "confusion": confusion,
        "audit": audit,
        "best_model": summarize_best_low_do_model(metrics),
        "literature_model_comparison": literature_model_comparison,
        "probability_thresholds": probability_thresholds,
        "permutation_importance": rf_permutation_importance,
        "threshold_sensitivity": threshold_sensitivity,
        "threshold_sensitivity_cv": threshold_sensitivity_cv,
    }


def run_low_do_grouped_cv(
    model_table: pd.DataFrame,
    predictors: list[str],
    models: dict[str, Pipeline] | None = None,
    output_path: Path | None = MODELING_DIR / "wqp_2021_2023_low_do_supervised_cv_metrics.csv",
) -> pd.DataFrame:
    x_values = model_table[predictors]
    y_values = model_table["target_low_do"]
    groups = model_table["MonitoringLocationIdentifier"]
    cv = StratifiedGroupKFold(n_splits=LOW_DO_CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for fold, (train_idx, test_idx) in enumerate(cv.split(x_values, y_values, groups), start=1):
        x_train, x_test = x_values.iloc[train_idx], x_values.iloc[test_idx]
        y_train, y_test = y_values.iloc[train_idx], y_values.iloc[test_idx]
        test_sites = model_table.iloc[test_idx]["MonitoringLocationIdentifier"]
        for model_name, model_template in make_low_do_models().items():
            model = model_template
            model.fit(x_train, y_train)
            predictions = model.predict(x_test)
            probabilities = get_positive_probabilities(model, x_test)
            row = evaluate_classifier(
                model_name,
                f"cv_fold_{fold}",
                y_test,
                predictions,
                probabilities,
                len(y_test),
                test_sites.nunique(),
            )
            row["fold"] = fold
            rows.append(row)
    cv_metrics = pd.DataFrame(rows)
    summary_rows = []
    metric_cols = [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "average_precision_pr_auc",
        "positive_rate_pct",
        "predicted_positive_rate_pct",
    ]
    for model_name, group in cv_metrics.groupby("model"):
        row = {"model": model_name, "split": "cv_mean", "fold": "mean"}
        row["fold_count"] = int(group["fold"].nunique())
        for col in metric_cols:
            row[col] = group[col].mean()
            row[f"{col}_std"] = group[col].std(ddof=0)
        row["row_count"] = group["row_count"].mean()
        row["unique_site_count"] = group["unique_site_count"].mean()
        summary_rows.append(row)
    cv_output = pd.concat([cv_metrics, pd.DataFrame(summary_rows)], ignore_index=True, sort=False)
    if output_path is not None:
        cv_output.to_csv(output_path, index=False)
    return cv_output


def write_low_do_interpretability_outputs(models: dict[str, Pipeline], predictors: list[str]) -> None:
    importance_rows = []
    if "logistic_regression" in models:
        logistic = models["logistic_regression"].named_steps["model"]
        coefficients = pd.DataFrame(
            {
                "model": "logistic_regression",
                "feature": predictors,
                "coefficient": logistic.coef_[0],
            }
        ).sort_values("coefficient", ascending=False)
        coefficients.to_csv(MODELING_DIR / "wqp_2021_2023_low_do_logistic_coefficients.csv", index=False)
        for _, row in coefficients.iterrows():
            importance_rows.append(
                {
                    "model": "logistic_regression",
                    "feature": row["feature"],
                    "importance_value": abs(row["coefficient"]),
                    "signed_value": row["coefficient"],
                    "importance_type": "absolute standardized coefficient",
                    "notes": "Coefficient describes model behavior, not causality.",
                }
            )
    for model_name in ["decision_tree", "gradient_boosting", "random_forest"]:
        model = models[model_name].named_steps["model"]
        for feature, importance in zip(predictors, model.feature_importances_):
            importance_rows.append(
                {
                    "model": model_name,
                    "feature": feature,
                    "importance_value": importance,
                    "signed_value": np.nan,
                    "importance_type": "feature_importance",
                    "notes": "Feature importance describes model behavior, not causality.",
                }
            )
    feature_importance = pd.DataFrame(importance_rows)
    feature_importance.to_csv(MODELING_DIR / "wqp_2021_2023_low_do_feature_importance.csv", index=False)

    tree_model = models["decision_tree"].named_steps["model"]
    tree_rules = export_text(tree_model, feature_names=predictors, max_depth=4)
    (MODELING_DIR / "wqp_2021_2023_low_do_decision_tree_rules.txt").write_text(
        "Project-defined low-DO condition decision tree rules. These rules describe model behavior, not causality.\n\n"
        + tree_rules,
        encoding="utf-8",
    )


def write_low_do_supervised_audit(
    training_summary: pd.DataFrame,
    predictors: list[str],
    train: pd.DataFrame,
    test: pd.DataFrame,
    overlap_sites: set[str],
) -> pd.DataFrame:
    row = training_summary.iloc[0]
    excluded_ids = [
        "MonitoringLocationIdentifier",
        "ResultIdentifier",
        "ActivityIdentifier",
        "source_row_key",
        "source_row_number",
    ]
    audit = pd.DataFrame(
        [
            {
                "target_name": "low_dissolved_oxygen",
                "threshold": row["threshold"],
                "threshold_source": row["threshold_source"],
                "design_level": row["design_level"],
                "split_strategy": row["split_strategy"],
                "leakage_predictors_excluded": "; ".join(sorted(LOW_DO_LEAKAGE_PREDICTORS)),
                "predictors_used": "; ".join(predictors),
                "id_fields_excluded": "; ".join(excluded_ids),
                "grouped_split_used": "yes",
                "unique_train_sites": train["MonitoringLocationIdentifier"].nunique(),
                "unique_test_sites": test["MonitoringLocationIdentifier"].nunique(),
                "overlap_in_train_test_sites": len(overlap_sites),
                "imputation_inside_pipeline": "yes",
                "notes": "Grouped-by-site split reduces site leakage; threshold is project-defined; no raw IDs or target-derived fields are predictors.",
            }
        ]
    )
    audit.to_csv(MODELING_DIR / "wqp_2021_2023_low_do_supervised_audit.csv", index=False)
    return audit


def summarize_best_low_do_model(metrics: pd.DataFrame) -> pd.DataFrame:
    test_metrics = metrics[metrics["split"].eq("test")].copy()
    best_f1 = test_metrics.sort_values(["f1", "average_precision_pr_auc"], ascending=False).iloc[0]
    best_pr = test_metrics.sort_values(["average_precision_pr_auc", "f1"], ascending=False).iloc[0]
    return pd.DataFrame(
        [
            {
                "criterion": "best_test_f1",
                "model": best_f1["model"],
                "f1": best_f1["f1"],
                "average_precision_pr_auc": best_f1["average_precision_pr_auc"],
                "balanced_accuracy": best_f1["balanced_accuracy"],
                "recall": best_f1["recall"],
                "precision": best_f1["precision"],
            },
            {
                "criterion": "best_test_pr_auc",
                "model": best_pr["model"],
                "f1": best_pr["f1"],
                "average_precision_pr_auc": best_pr["average_precision_pr_auc"],
                "balanced_accuracy": best_pr["balanced_accuracy"],
                "recall": best_pr["recall"],
                "precision": best_pr["precision"],
            },
        ]
    )


def write_literature_aligned_model_comparison(metrics: pd.DataFrame, cv_metrics: pd.DataFrame) -> pd.DataFrame:
    print("Writing literature-aligned supervised model comparison")
    test_metrics = metrics[metrics["split"].eq("test")].copy()
    cv_mean = cv_metrics[cv_metrics["split"].eq("cv_mean")].copy()
    best_f1_model = test_metrics.sort_values(["f1", "average_precision_pr_auc"], ascending=False).iloc[0]["model"]
    best_pr_model = test_metrics.sort_values(["average_precision_pr_auc", "f1"], ascending=False).iloc[0]["model"]
    model_families = {
        "majority_class_baseline": "baseline",
        "logistic_regression": "linear",
        "decision_tree": "single decision tree",
        "random_forest": "bagged tree ensemble",
        "gradient_boosting": "boosted tree ensemble",
    }
    literature_notes = {
        "majority_class_baseline": "Class-balance baseline for interpreting supervised metrics.",
        "logistic_regression": "Simple linear supervised baseline.",
        "decision_tree": "Simple tree baseline related to interpretable tree models.",
        "random_forest": "Tree-based ensemble commonly used in water-quality ML studies.",
        "gradient_boosting": "Boosted-tree ensemble commonly used in water-quality and DO/hypoxia ML studies.",
    }
    merged = test_metrics.merge(
        cv_mean[["model", "f1", "average_precision_pr_auc"]],
        on="model",
        how="left",
        suffixes=("", "_cv_mean"),
    )
    rows = []
    for _, row in merged.iterrows():
        model_name = row["model"]
        selected = model_name == best_f1_model
        if model_name == best_f1_model and model_name == best_pr_model:
            reason = "Best by both held-out test F1 and PR-AUC."
        elif model_name == best_f1_model:
            reason = "Best by held-out test F1; compare PR-AUC before changing presentation focus."
        elif model_name == best_pr_model:
            reason = "Best by held-out test PR-AUC; compare F1 before changing presentation focus."
        elif model_name == "gradient_boosting":
            reason = "Literature-aligned boosted-tree comparison; not selected unless it improves the main held-out and CV metrics."
        else:
            reason = "Comparison model retained for baseline context."
        rows.append(
            {
                "model": model_name,
                "model_display_name": display_model_name(model_name),
                "model_family": model_families.get(model_name, "other"),
                "literature_alignment_note": literature_notes.get(model_name, ""),
                "test_f1": row["f1"],
                "test_pr_auc": row["average_precision_pr_auc"],
                "test_roc_auc": row["roc_auc"],
                "test_balanced_accuracy": row["balanced_accuracy"],
                "test_precision": row["precision"],
                "test_recall": row["recall"],
                "cv_mean_f1": row["f1_cv_mean"],
                "cv_mean_pr_auc": row["average_precision_pr_auc_cv_mean"],
                "selected_as_best": "yes" if selected else "no",
                "reason": reason,
            }
        )
    comparison = pd.DataFrame(rows).sort_values(["test_f1", "test_pr_auc"], ascending=False)
    comparison.to_csv(
        MODELING_DIR / "wqp_2021_2023_low_do_literature_aligned_model_comparison.csv",
        index=False,
    )
    return comparison


def write_low_do_rf_probability_threshold_analysis(
    rf_model: Pipeline,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    print("Writing random forest probability-threshold analysis")
    probabilities = get_positive_probabilities(rf_model, x_test)
    rows = []
    for threshold in LOW_DO_RF_PROBABILITY_THRESHOLDS:
        predictions = (probabilities >= threshold).astype(int)
        cm = confusion_matrix(y_test, predictions, labels=[0, 1])
        rows.append(
            {
                "model": "random_forest",
                "target_threshold": "median dissolved oxygen < 5 mg/L",
                "probability_threshold": threshold,
                "row_count": len(y_test),
                "actual_positive_rate_pct": float(np.mean(y_test) * 100),
                "predicted_positive_rate_pct": float(np.mean(predictions) * 100),
                "precision": float(precision_score(y_test, predictions, zero_division=0)),
                "recall": float(recall_score(y_test, predictions, zero_division=0)),
                "f1": float(f1_score(y_test, predictions, zero_division=0)),
                "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
                "false_positives": int(cm[0, 1]),
                "false_negatives": int(cm[1, 0]),
                "true_positives": int(cm[1, 1]),
                "true_negatives": int(cm[0, 0]),
                "notes": "Held-out grouped test set; threshold tradeoffs describe screening behavior, not operational deployment.",
            }
        )
    threshold_analysis = pd.DataFrame(rows)
    threshold_analysis.to_csv(
        MODELING_DIR / "wqp_2021_2023_low_do_rf_probability_threshold_analysis.csv",
        index=False,
    )

    plot_data = threshold_analysis.melt(
        id_vars="probability_threshold",
        value_vars=["precision", "recall", "f1", "balanced_accuracy"],
        var_name="metric",
        value_name="score",
    )
    plt.figure(figsize=(8.5, 5))
    sns.lineplot(data=plot_data, x="probability_threshold", y="score", hue="metric", marker="o")
    plt.title("2021-2023 Project-Defined Low-DO RF\nPrecision-Recall Threshold Tradeoff")
    plt.xlabel("Random forest probability threshold")
    plt.ylabel("Held-out test score")
    plt.ylim(0, 1)
    plt.legend(title="Metric", bbox_to_anchor=(1.02, 1), loc="upper left")
    save_current_figure(MODELING_FIG_DIR / "wqp_2021_2023_low_do_precision_recall_thresholds.png")
    return threshold_analysis


def write_low_do_permutation_importance(
    rf_model: Pipeline,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    predictors: list[str],
) -> pd.DataFrame:
    print("Writing random forest permutation importance")
    result = permutation_importance(
        rf_model,
        x_test,
        y_test,
        scoring="average_precision",
        n_repeats=LOW_DO_PERMUTATION_REPEATS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    built_in = rf_model.named_steps["model"].feature_importances_
    importance = pd.DataFrame(
        {
            "model": "random_forest",
            "feature": predictors,
            "scoring": "average_precision",
            "n_repeats": LOW_DO_PERMUTATION_REPEATS,
            "permutation_importance_mean": result.importances_mean,
            "permutation_importance_std": result.importances_std,
            "built_in_feature_importance": built_in,
            "notes": "Permutation and built-in importance describe model behavior on held-out data, not causality.",
        }
    ).sort_values("permutation_importance_mean", ascending=False)
    importance.to_csv(MODELING_DIR / "wqp_2021_2023_low_do_permutation_importance.csv", index=False)

    plt.figure(figsize=(8, 5))
    sns.barplot(data=importance, x="permutation_importance_mean", y="feature", color="#59A14F")
    plt.errorbar(
        x=importance["permutation_importance_mean"],
        y=np.arange(len(importance)),
        xerr=importance["permutation_importance_std"],
        fmt="none",
        ecolor="#333333",
        capsize=3,
    )
    plt.title("2021-2023 Project-Defined Low-DO RF\nPermutation Importance")
    plt.xlabel("Mean decrease in average precision")
    plt.ylabel("Predictor")
    save_current_figure(MODELING_FIG_DIR / "wqp_2021_2023_low_do_permutation_importance.png")
    return importance


def run_low_do_threshold_sensitivity(
    low_do_readiness: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("Running low-DO threshold sensitivity checks")
    design_table = low_do_readiness["design_tables"][LOW_DO_TRAINING_DESIGN].copy()
    predictors = list(LOW_DO_PREDICTOR_UNITS)
    summary_rows = []
    cv_outputs = []

    for threshold in LOW_DO_THRESHOLDS:
        model_table = build_low_do_model_table_for_threshold(design_table, threshold)
        train_idx, test_idx = grouped_train_test_split(model_table)
        train = model_table.iloc[train_idx].copy()
        test = model_table.iloc[test_idx].copy()
        x_train = train[predictors]
        y_train = train["target_low_do"]
        x_test = test[predictors]
        y_test = test["target_low_do"]

        test_rows = []
        for model_name, model in make_low_do_models().items():
            model.fit(x_train, y_train)
            predictions = model.predict(x_test)
            probabilities = get_positive_probabilities(model, x_test)
            row = evaluate_classifier(
                model_name,
                "test",
                y_test,
                predictions,
                probabilities,
                len(y_test),
                test["MonitoringLocationIdentifier"].nunique(),
            )
            test_rows.append(row)
        test_metrics = pd.DataFrame(test_rows)
        best_f1 = test_metrics.sort_values(["f1", "average_precision_pr_auc"], ascending=False).iloc[0]
        best_pr = test_metrics.sort_values(["average_precision_pr_auc", "f1"], ascending=False).iloc[0]
        rf_row = test_metrics[test_metrics["model"].eq("random_forest")].iloc[0]

        cv_metrics = run_low_do_grouped_cv(model_table, predictors, output_path=None)
        cv_metrics.insert(0, "threshold_value_mg_L", threshold)
        cv_metrics.insert(1, "target_definition", f"median dissolved oxygen < {threshold:g} mg/L")
        cv_outputs.append(cv_metrics)
        rf_cv = cv_metrics[
            cv_metrics["model"].eq("random_forest") & cv_metrics["split"].eq("cv_mean")
        ].iloc[0]

        positives = int(model_table["target_low_do"].sum())
        total = len(model_table)
        summary_rows.append(
            {
                "threshold_value_mg_L": threshold,
                "target_definition": f"median dissolved oxygen < {threshold:g} mg/L",
                "threshold_source": "project-defined screening threshold",
                "design_level": LOW_DO_TRAINING_DESIGN,
                "row_count": total,
                "positive_count": positives,
                "negative_count": total - positives,
                "positive_rate_pct": positives / total * 100 if total else 0.0,
                "best_model_by_f1": best_f1["model"],
                "best_model_by_pr_auc": best_pr["model"],
                "random_forest_f1": rf_row["f1"],
                "random_forest_precision": rf_row["precision"],
                "random_forest_recall": rf_row["recall"],
                "random_forest_roc_auc": rf_row["roc_auc"],
                "random_forest_pr_auc": rf_row["average_precision_pr_auc"],
                "random_forest_grouped_cv_mean_f1": rf_cv["f1"],
                "random_forest_grouped_cv_mean_pr_auc": rf_cv["average_precision_pr_auc"],
                "notes": "Sensitivity only; the main project result remains the <5 mg/L target unless reporting explains the tradeoff.",
            }
        )

    sensitivity = pd.DataFrame(summary_rows)
    sensitivity.to_csv(
        MODELING_DIR / "wqp_2021_2023_low_do_threshold_sensitivity_metrics.csv",
        index=False,
    )
    sensitivity_cv = pd.concat(cv_outputs, ignore_index=True, sort=False)
    sensitivity_cv.to_csv(
        MODELING_DIR / "wqp_2021_2023_low_do_threshold_sensitivity_cv_metrics.csv",
        index=False,
    )

    plot_data = sensitivity.copy()
    plot_data["threshold_label"] = plot_data["threshold_value_mg_L"].map(lambda x: f"< {x:g} mg/L")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.barplot(data=plot_data, x="threshold_label", y="positive_rate_pct", color="#4C78A8", ax=axes[0])
    axes[0].set_title("Target Positive Rate")
    axes[0].set_xlabel("Low-DO threshold")
    axes[0].set_ylabel("Positive rate (%)")

    metric_plot = plot_data.melt(
        id_vars="threshold_label",
        value_vars=[
            "random_forest_precision",
            "random_forest_recall",
            "random_forest_f1",
            "random_forest_pr_auc",
        ],
        var_name="metric",
        value_name="score",
    )
    sns.lineplot(data=metric_plot, x="threshold_label", y="score", hue="metric", marker="o", ax=axes[1])
    axes[1].set_title("Random Forest Test Metrics")
    axes[1].set_xlabel("Low-DO threshold")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0, 1)
    axes[1].legend(title="Metric", bbox_to_anchor=(1.02, 1), loc="upper left")
    save_current_figure(MODELING_FIG_DIR / "wqp_2021_2023_low_do_threshold_sensitivity.png")
    return sensitivity, sensitivity_cv


def plot_low_do_supervised_figures(metrics: pd.DataFrame, cv_metrics: pd.DataFrame, confusion: pd.DataFrame) -> None:
    test_metrics = metrics[metrics["split"].eq("test")].copy()
    test_metrics["model_label"] = test_metrics["model"].map(display_model_name)
    metric_plot = test_metrics.melt(
        id_vars="model_label",
        value_vars=["balanced_accuracy", "precision", "recall", "f1", "average_precision_pr_auc"],
        var_name="metric",
        value_name="value",
    )
    plt.figure(figsize=(11, 5.8))
    sns.barplot(data=metric_plot, x="metric", y="value", hue="model_label")
    plt.title("2021-2023 Project-Defined Low-DO Condition: Test Metric Comparison")
    plt.xlabel("Metric")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=25, ha="right")
    plt.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
    save_current_figure(MODELING_FIG_DIR / "wqp_2021_2023_low_do_model_metric_comparison.png")

    cv_mean = cv_metrics[cv_metrics["split"].eq("cv_mean")].copy()
    cv_mean["model_label"] = cv_mean["model"].map(display_model_name)
    cv_plot = cv_mean.melt(
        id_vars="model_label",
        value_vars=["balanced_accuracy", "f1", "average_precision_pr_auc"],
        var_name="metric",
        value_name="value",
    )
    plt.figure(figsize=(10, 5.3))
    sns.barplot(data=cv_plot, x="metric", y="value", hue="model_label")
    plt.title("2021-2023 Project-Defined Low-DO Condition: Grouped CV Summary")
    plt.xlabel("Metric")
    plt.ylabel("Mean CV score")
    plt.ylim(0, 1)
    plt.xticks(rotation=20, ha="right")
    plt.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
    save_current_figure(MODELING_FIG_DIR / "wqp_2021_2023_low_do_cv_metric_summary.png")

    test_confusion = confusion[confusion["split"].eq("test")].copy()
    best_model = test_metrics.sort_values(["f1", "average_precision_pr_auc"], ascending=False).iloc[0]["model"]
    cm_row = test_confusion[test_confusion["model"].eq(best_model)].iloc[0]
    cm = np.array([[cm_row["true_negative"], cm_row["false_positive"]], [cm_row["false_negative"], cm_row["true_positive"]]])
    plt.figure(figsize=(5.8, 4.8))
    sns.heatmap(cm, annot=True, fmt=".0f", cmap="Blues", cbar=False, xticklabels=["Predicted non-low", "Predicted low"], yticklabels=["Actual non-low", "Actual low"])
    plt.title(f"2021-2023 Project-Defined Low-DO Condition\nConfusion Matrix: {display_model_name(best_model)}")
    plt.xlabel("Predicted class")
    plt.ylabel("Actual class")
    save_current_figure(MODELING_FIG_DIR / "wqp_2021_2023_low_do_confusion_matrix.png")

    importance = pd.read_csv(MODELING_DIR / "wqp_2021_2023_low_do_feature_importance.csv")
    rf_importance = importance[importance["model"].eq("random_forest")].sort_values("importance_value", ascending=False)
    plt.figure(figsize=(8, 5))
    sns.barplot(data=rf_importance, x="importance_value", y="feature", color="#4C78A8")
    plt.title("2021-2023 Project-Defined Low-DO Condition\nRandom Forest Feature Importance")
    plt.xlabel("Feature importance")
    plt.ylabel("Predictor")
    save_current_figure(MODELING_FIG_DIR / "wqp_2021_2023_low_do_feature_importance.png")


def print_modeling_summary(
    core_feature_matrix: pd.DataFrame,
    expanded_feature_matrix: pd.DataFrame | None,
    core_explained: pd.DataFrame,
    expanded_explained: pd.DataFrame | None,
    cluster_scores: pd.DataFrame,
    chosen_k: int,
    chosen_reason: str,
    low_do_readiness: dict[str, pd.DataFrame],
    supervised_results: dict[str, pd.DataFrame | str],
) -> None:
    """Print a concise command-line summary without writing markdown reports."""
    core_pc1 = core_explained.iloc[0]["explained_variance_ratio"] * 100
    core_pc2 = core_explained.iloc[1]["explained_variance_ratio"] * 100 if len(core_explained) > 1 else 0
    print("2021-2023 first-phase modeling complete.")
    print(f"Core feature matrix: {core_feature_matrix.shape[0]:,} sites x {len(CORE_VARIABLES)} variables")
    if expanded_feature_matrix is not None:
        print(f"Expanded feature matrix: {expanded_feature_matrix.shape[0]:,} sites x {len(EXPANDED_VARIABLES)} variables")
        if expanded_explained is not None and not expanded_explained.empty:
            expanded_pc1 = expanded_explained.iloc[0]["explained_variance_ratio"] * 100
            print(f"Expanded PCA PC1 explained variance: {expanded_pc1:.1f}%")
    else:
        print("Expanded feature matrix: not created")
    print(f"Core PCA explained variance: PC1={core_pc1:.1f}%, PC2={core_pc2:.1f}%")

    chosen_row = cluster_scores.loc[cluster_scores["k"].eq(chosen_k)].iloc[0]
    print(f"Selected clustering k: {chosen_k} (silhouette={chosen_row['silhouette_score']:.3f}; {chosen_reason})")

    training_summary = supervised_results["training_summary"]
    summary_row = training_summary.iloc[0]
    print(
        "Low-DO supervised table: "
        f"{int(summary_row['rows_after_filter']):,} rows, "
        f"{int(summary_row['unique_sites']):,} sites, "
        f"positive rate {summary_row['positive_rate_pct']:.1f}%"
    )
    readiness_row = low_do_readiness["recommendation"].iloc[0]
    print(
        "Low-DO target: "
        f"{readiness_row['recommended_threshold']} at {readiness_row['recommended_design_level']} "
        f"with {readiness_row['split_strategy']}"
    )

    best = supervised_results["best_model"]
    metrics = supervised_results["metrics"]
    best_f1_model = best.loc[best["criterion"].eq("best_test_f1"), "model"].iloc[0]
    best_pr_model = best.loc[best["criterion"].eq("best_test_pr_auc"), "model"].iloc[0]
    best_row = metrics[metrics["split"].eq("test") & metrics["model"].eq(best_f1_model)].iloc[0]
    print(
        "Best supervised model by test F1: "
        f"{best_f1_model} (F1={best_row['f1']:.3f}, "
        f"PR-AUC={best_row['average_precision_pr_auc']:.3f}, "
        f"ROC-AUC={best_row['roc_auc']:.3f}, recall={best_row['recall']:.3f}, "
        f"precision={best_row['precision']:.3f})"
    )
    print(f"Best supervised model by test PR-AUC: {best_pr_model}")
    print(f"Wrote modeling CSVs to {MODELING_DIR}")
    print(f"Wrote modeling figures to {MODELING_FIG_DIR}")
def main() -> None:
    ensure_dirs()
    inputs = load_inputs()
    input_audit = write_modeling_input_audit(inputs)
    variable_audit = audit_selected_variables(inputs)

    site_matrix = inputs["site_matrix"]
    site_summary = inputs["site_summary"]
    core_feature, core_raw_values, core_imputed, core_standardized, core_missing = build_feature_matrix(
        "core", site_matrix, site_summary, CORE_VARIABLES, MIN_CORE_MEASURED
    )
    expanded_feature = None
    expanded_raw_values = None
    expanded_imputed = None
    expanded_standardized = None
    expanded_missing = pd.DataFrame()
    if all(var in site_matrix.columns for var in EXPANDED_VARIABLES):
        expanded_feature, expanded_raw_values, expanded_imputed, expanded_standardized, expanded_missing = build_feature_matrix(
            "expanded", site_matrix, site_summary, EXPANDED_VARIABLES, MIN_EXPANDED_MEASURED
        )
        if len(expanded_feature) < MIN_EXPANDED_SITES:
            print("Expanded matrix has too few eligible sites; keeping file for audit but not using it as primary model.")

    missing_summary = pd.concat([core_missing, expanded_missing], ignore_index=True, sort=False)
    missing_summary.to_csv(MODELING_DIR / "wqp_2021_2023_missingness_imputation_summary.csv", index=False)

    write_spearman_outputs("core", core_raw_values, CORE_VARIABLES)
    core_scores, core_loadings, core_explained = run_pca("core", core_feature, core_standardized, CORE_VARIABLES)

    expanded_explained = None
    if expanded_feature is not None and expanded_raw_values is not None and expanded_standardized is not None:
        write_spearman_outputs("expanded", expanded_raw_values, EXPANDED_VARIABLES)
        _, _, expanded_explained = run_pca("expanded", expanded_feature, expanded_standardized, EXPANDED_VARIABLES)

    cluster_scores, assignments, cluster_profiles, metadata_summary, chosen_k, chosen_reason = run_clustering(
        "core", core_feature, core_standardized, core_scores, CORE_VARIABLES
    )
    target_feasibility, temporal_readiness = scan_numeric_long()
    low_do_readiness = run_low_do_supervised_readiness()
    supervised_results = train_low_do_supervised_models(low_do_readiness)
    print_modeling_summary(
        core_feature,
        expanded_feature,
        core_explained,
        expanded_explained,
        cluster_scores,
        chosen_k,
        chosen_reason,
        low_do_readiness,
        supervised_results,
    )


if __name__ == "__main__":
    main()

