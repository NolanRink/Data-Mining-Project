"""
Florida Water Quality Portal data exploration.

This script profiles the raw WQP resultphyschem export and station metadata
without modifying the raw files. It writes reproducible EDA tables, figures,
derived analysis-ready extracts, and validation tables used to choose realistic
data-mining techniques.
"""

from __future__ import annotations

import math
import textwrap
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

try:
    import seaborn as sns
except ImportError:  # pragma: no cover - seaborn is optional
    sns = None


BASE_DIR = Path(__file__).resolve().parents[1]
RESULT_PATH = BASE_DIR / "data" / "resultphyschem.csv"
STATION_PATH = BASE_DIR / "data" / "station.csv"
EDA_DIR = BASE_DIR / "outputs" / "eda"
FIG_DIR = BASE_DIR / "outputs" / "figures"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

CHUNKSIZE = 100_000
UNIQUE_CAP = 250_000
EXAMPLE_CAP = 5
MIN_CANDIDATE_ROWS = 5_000
MIN_CANDIDATE_SITES = 50
EXPECTED_BASELINE_YEARS = {2021, 2022, 2023}
MIN_ACTIVE_YEARS_FOR_CONTINUITY = len(EXPECTED_BASELINE_YEARS)
MATRIX_MIN_OBS_PER_SITE_CHAR = 3
LOW_RECORD_COUNT_THRESHOLD = 10
LOW_CHARACTERISTIC_COUNT_THRESHOLD = 3
FIG_DPI = 240


def wrap_label(value: object, width: int = 24) -> str:
    text = str(value)
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False)) or text


def save_current_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()


def format_count_axis(axis: str = "x") -> None:
    formatter = mticker.StrMethodFormatter("{x:,.0f}")
    ax = plt.gca()
    if axis == "x":
        ax.xaxis.set_major_formatter(formatter)
    else:
        ax.yaxis.set_major_formatter(formatter)
LIMITED_MONTH_COVERAGE_THRESHOLD = 3
MISSING_UNIT_LABEL = "Missing unit"
UNITLESS_LABEL = "Unitless"


CORE_COLUMNS = [
    "OrganizationIdentifier",
    "OrganizationFormalName",
    "ActivityIdentifier",
    "ActivityTypeCode",
    "ActivityMediaName",
    "ActivityMediaSubdivisionName",
    "ActivityStartDate",
    "ActivityStartTime/Time",
    "MonitoringLocationIdentifier",
    "MonitoringLocationName",
    "ActivityLocation/LatitudeMeasure",
    "ActivityLocation/LongitudeMeasure",
    "ResultIdentifier",
    "ResultDetectionConditionText",
    "CharacteristicName",
    "ResultSampleFractionText",
    "ResultMeasureValue",
    "ResultMeasure/MeasureUnitCode",
    "MeasureQualifierCode",
    "ResultStatusIdentifier",
    "ResultValueTypeName",
    "USGSPCode",
    "DetectionQuantitationLimitTypeName",
    "DetectionQuantitationLimitMeasure/MeasureValue",
    "DetectionQuantitationLimitMeasure/MeasureUnitCode",
    "ResultCommentText",
    "LastUpdated",
    "ProviderName",
]

LIKELY_KEY_COLUMNS = [
    "OrganizationIdentifier",
    "ActivityIdentifier",
    "ResultIdentifier",
    "MonitoringLocationIdentifier",
    "ActivityStartDate",
    "CharacteristicName",
    "ResultMeasureValue",
    "ResultMeasure/MeasureUnitCode",
]

COMMON_VARIABLE_PATTERNS = {
    "Temperature": ["temperature"],
    "pH": ["ph"],
    "Dissolved oxygen": ["dissolved oxygen", "oxygen, dissolved"],
    "Specific conductance / conductivity": ["specific conductance", "conductivity"],
    "Turbidity": ["turbidity"],
    "Nitrogen / nitrate / nitrite / ammonia": [
        "nitrogen",
        "nitrate",
        "nitrite",
        "ammonia",
        "kjeldahl",
    ],
    "Phosphorus / orthophosphate": ["phosphorus", "orthophosphate", "phosphate"],
    "Chlorophyll": ["chlorophyll"],
    "E. coli or fecal coliform": ["e. coli", "escherichia", "fecal coliform"],
    "Salinity / chloride": ["salinity", "chloride"],
    "Metals": [
        "arsenic",
        "cadmium",
        "chromium",
        "copper",
        "iron",
        "lead",
        "manganese",
        "mercury",
        "nickel",
        "selenium",
        "silver",
        "zinc",
        "aluminum",
    ],
    "Cyanobacteria/cyanotoxin": [
        "cyanobacteria",
        "cyanotoxin",
        "microcystin",
        "anatoxin",
        "cylindrospermopsin",
        "saxitoxin",
        "phycocyanin",
    ],
}


def ensure_dirs() -> None:
    for path in [EDA_DIR, FIG_DIR, PROCESSED_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def bytes_to_mb(value: int | float) -> float:
    return round(value / 1024**2, 2)


def normalize_text(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def normalize_unit(unit_series: pd.Series, characteristic_series: pd.Series | None = None) -> pd.Series:
    """Normalize result units consistently across summaries and matrix construction."""
    units = unit_series.astype("string").str.strip()
    units = units.mask(units.isna() | units.eq(""), MISSING_UNIT_LABEL)
    if characteristic_series is not None:
        chars = characteristic_series.astype("string").str.strip().str.lower()
        ph_mask = chars.eq("ph") & units.eq(MISSING_UNIT_LABEL)
        units = units.mask(ph_mask, UNITLESS_LABEL)
    return units


def season_from_month(month: float) -> str | float:
    if pd.isna(month):
        return np.nan
    month = int(month)
    if month in (12, 1, 2):
        return "Winter"
    if month in (3, 4, 5):
        return "Spring"
    if month in (6, 7, 8):
        return "Summer"
    return "Fall"


def classify_column(name: str) -> str:
    lower = name.lower()
    if "identifier" in lower or lower.endswith("id") or "usgspcode" in lower:
        return "ID"
    if "date" in lower or "time" in lower or "lastupdated" in lower:
        return "date/time"
    if "latitude" in lower or "longitude" in lower or "huc" in lower or "county" in lower:
        return "geographic field"
    if "measurevalue" in lower or lower.endswith("value") or "numeric" in lower:
        return "numeric measurement"
    if "unit" in lower:
        return "unit field"
    if any(token in lower for token in ["quality", "qualifier", "status", "detection", "limit"]):
        return "quality-control field"
    if any(token in lower for token in ["name", "type", "code", "provider", "organization", "characteristic"]):
        return "categorical variable"
    return "metadata field"


def useful_column_reason(name: str, missing_pct: float, unique_count: str) -> tuple[str, str]:
    lower = name.lower()
    high_value_tokens = [
        "organizationidentifier",
        "activityidentifier",
        "activitystartdate",
        "monitoringlocationidentifier",
        "monitoringlocationname",
        "latitude",
        "longitude",
        "resultidentifier",
        "resultdetectionconditiontext",
        "characteristicname",
        "resultsamplefractiontext",
        "resultmeasurevalue",
        "resultmeasure/measureunitcode",
        "measurequalifiercode",
        "resultstatusidentifier",
        "detectionquantitationlimittype",
        "detectionquantitationlimitmeasure/measurevalue",
        "providername",
    ]
    if any(token in lower for token in high_value_tokens):
        return "Yes", "Core field for keys, dates, sites, measurements, units, geography, or quality checks."
    if missing_pct >= 98:
        return "No", "Nearly empty in this export."
    if "comment" in lower or "description" in lower or "url" in lower:
        return "Limited", "Mostly audit text; useful for QA but not primary modeling."
    if "identifier" in lower and unique_count not in {"0", "1"}:
        return "Limited", "Identifier is useful for joins/deduplication but should not be a model feature."
    return "Possible", "May be useful depending on the final analysis level."


def classify_characteristic_group(name: object) -> str:
    if pd.isna(name):
        return "Unknown"
    lower = str(name).lower()
    for group in COMMON_VARIABLE_PATTERNS:
        if matches_common_variable(group, lower):
            return group
    if any(token in lower for token in ["calcium", "magnesium", "sodium", "potassium", "sulfate", "alkalinity"]):
        return "Major ions / hardness / alkalinity"
    if any(token in lower for token in ["carbon", "organic", "toc", "doc"]):
        return "Carbon / organic matter"
    if any(token in lower for token in ["bacteria", "enterococcus", "coliform"]):
        return "Microbiological"
    if any(token in lower for token in ["depth", "gage", "flow", "discharge"]):
        return "Hydrologic / field context"
    return "Other"


def matches_common_variable(group: str, value: str) -> bool:
    value = str(value).lower().strip()
    if group == "pH":
        return value == "ph" or value.startswith("ph,") or value.startswith("ph ") or value.startswith("ph(")
    return any(token in value for token in COMMON_VARIABLE_PATTERNS[group])


def summarize_common_variable(name: str, characteristic_summary: pd.DataFrame) -> dict[str, object]:
    mask = characteristic_summary["CharacteristicName"].astype(str).str.lower().apply(
        lambda value: matches_common_variable(name, value)
    )
    sub = characteristic_summary.loc[mask]
    if sub.empty:
        return {
            "common_variable": name,
            "present": False,
            "matching_characteristics": "",
            "record_count": 0,
            "numeric_count": 0,
            "site_count": 0,
            "active_years": 0,
            "top_units": "",
        }
    return {
        "common_variable": name,
        "present": True,
        "matching_characteristics": "; ".join(sub["CharacteristicName"].head(10).astype(str).tolist()),
        "record_count": int(sub["record_count"].sum()),
        "numeric_count": int(sub["numeric_count"].sum()),
        "site_count": int(sub["site_count"].max()),
        "active_years": int(sub["active_years"].max()),
        "top_units": "; ".join(sub["top_units"].head(5).astype(str).tolist()),
    }


def read_station_metadata() -> pd.DataFrame:
    station = pd.read_csv(STATION_PATH, low_memory=False)
    station["LatitudeMeasure"] = pd.to_numeric(station.get("LatitudeMeasure"), errors="coerce")
    station["LongitudeMeasure"] = pd.to_numeric(station.get("LongitudeMeasure"), errors="coerce")
    keep = [
        "MonitoringLocationIdentifier",
        "MonitoringLocationName",
        "MonitoringLocationTypeName",
        "HUCEightDigitCode",
        "LatitudeMeasure",
        "LongitudeMeasure",
        "CountyCode",
        "ProviderName",
        "OrganizationIdentifier",
    ]
    keep = [col for col in keep if col in station.columns]
    station = station[keep].drop_duplicates("MonitoringLocationIdentifier")
    return station


def safe_value_counts(counter: Counter, n: int = 10) -> str:
    return "; ".join(f"{key}: {value:,}" for key, value in counter.most_common(n))


def save_bar(data: pd.DataFrame, x: str, y: str, path: Path, title: str, xlabel: str, ylabel: str, rotate: int = 0) -> None:
    if data.empty:
        return
    plt.figure(figsize=(12, 6.5))
    if sns is not None:
        sns.barplot(data=data, x=x, y=y, color="#4C78A8")
    else:
        plt.bar(data[x].astype(str), data[y], color="#4C78A8")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    format_count_axis("y" if y == "record_count" else "x")
    plt.grid(axis="y", alpha=0.2)
    if rotate:
        plt.xticks(rotation=rotate, ha="right", fontsize=9)
    save_current_figure(path)


def save_horizontal_bar(data: pd.DataFrame, label: str, value: str, path: Path, title: str, xlabel: str, ylabel: str, width: int = 28) -> None:
    if data.empty:
        return
    plot_data = data.copy()
    plot_data[label] = plot_data[label].map(lambda item: wrap_label(item, width))
    plt.figure(figsize=(12, max(6, len(plot_data) * 0.35)))
    if sns is not None:
        sns.barplot(data=plot_data, x=value, y=label, color="#4C78A8")
    else:
        plt.barh(plot_data[label].astype(str), plot_data[value], color="#4C78A8")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    format_count_axis("x")
    plt.grid(axis="x", alpha=0.2)
    save_current_figure(path)


def summarize_counter(counter: Counter, total: int, name_col: str) -> pd.DataFrame:
    rows = []
    for key, count in counter.most_common():
        rows.append({name_col: key, "record_count": count, "record_pct": count / total * 100 if total else np.nan})
    return pd.DataFrame(rows)


def build_profile() -> dict[str, object]:
    ensure_dirs()
    station = read_station_metadata()
    site_type_map = station.set_index("MonitoringLocationIdentifier").get("MonitoringLocationTypeName", pd.Series(dtype="object")).to_dict()
    station_lat_map = station.set_index("MonitoringLocationIdentifier").get("LatitudeMeasure", pd.Series(dtype="float")).to_dict()
    station_lon_map = station.set_index("MonitoringLocationIdentifier").get("LongitudeMeasure", pd.Series(dtype="float")).to_dict()
    county_map = station.set_index("MonitoringLocationIdentifier").get("CountyCode", pd.Series(dtype="object")).to_dict()
    huc_map = station.set_index("MonitoringLocationIdentifier").get("HUCEightDigitCode", pd.Series(dtype="object")).to_dict()

    header = pd.read_csv(RESULT_PATH, nrows=0).columns.tolist()
    sample = pd.read_csv(RESULT_PATH, nrows=10_000, low_memory=False)
    sample_memory_mb = sample.memory_usage(deep=True).sum() / 1024**2
    estimated_memory_mb = sample_memory_mb / max(len(sample), 1) * estimate_row_count_fast() if len(sample) else np.nan

    missing_counts = pd.Series(0, index=header, dtype="int64")
    inferred_dtypes: dict[str, set[str]] = {col: set() for col in header}
    unique_values: dict[str, set[object]] = {col: set() for col in header}
    unique_capped: dict[str, bool] = {col: False for col in header}
    examples: dict[str, list[str]] = {col: [] for col in header}

    total_rows = 0
    duplicate_full_hashes = Counter()
    duplicate_key_hashes = Counter()
    result_id_counter = Counter()

    provider_counter = Counter()
    organization_counter = Counter()
    site_type_record_counter = Counter()

    rows_by_year = Counter()
    rows_by_month = Counter()
    rows_by_year_month = Counter()
    rows_by_season = Counter()

    invalid_latlon = 0
    florida_latlon_outside = 0
    date_parse_failures = 0
    non_numeric_results = 0
    numeric_result_count = 0
    negative_numeric_count = 0
    less_than_text_count = 0
    result_measure_contains_lt = 0
    date_min = None
    date_max = None

    char_aggs: dict[str, dict[str, object]] = defaultdict(
        lambda: {
            "record_count": 0,
            "numeric_count": 0,
            "missing_result_count": 0,
            "negative_count": 0,
            "non_detect_count": 0,
            "sites": set(),
            "providers": Counter(),
            "organizations": Counter(),
            "units": Counter(),
            "years": set(),
            "min_date": None,
            "max_date": None,
            "min_value": math.inf,
            "max_value": -math.inf,
        }
    )
    unit_summary_counter = Counter()
    site_type_char_counter = Counter()

    site_aggs: dict[str, dict[str, object]] = defaultdict(
        lambda: {
            "record_count": 0,
            "numeric_count": 0,
            "characteristics": set(),
            "providers": Counter(),
            "organizations": Counter(),
            "years": set(),
            "months": set(),
            "min_date": None,
            "max_date": None,
            "lat_values": [],
            "lon_values": [],
        }
    )
    cleaned_path = PROCESSED_DIR / "florida_wqp_numeric_long.csv"
    cleaned_tmp_path = cleaned_path.with_suffix(".tmp.csv")
    if cleaned_tmp_path.exists():
        cleaned_tmp_path.unlink()
    cleaned_written = False
    selected_cols = [col for col in CORE_COLUMNS if col in header]
    processed_rows = 0
    processed_numeric_rows = 0
    processed_result_id_counter = Counter()

    for chunk in pd.read_csv(RESULT_PATH, chunksize=CHUNKSIZE, low_memory=False):
        chunk_rows = len(chunk)
        source_start = total_rows + 1
        source_row_number = np.arange(source_start, source_start + chunk_rows, dtype=np.int64)
        total_rows += chunk_rows
        missing_counts = missing_counts.add(chunk.isna().sum(), fill_value=0).astype("int64")

        for col in header:
            inferred_dtypes[col].add(str(chunk[col].dtype))
            non_null = chunk[col].dropna()
            if len(examples[col]) < EXAMPLE_CAP and not non_null.empty:
                for value in non_null.astype(str).head(EXAMPLE_CAP).tolist():
                    if value not in examples[col]:
                        examples[col].append(value)
                    if len(examples[col]) >= EXAMPLE_CAP:
                        break
            if not unique_capped[col]:
                values = pd.unique(non_null)
                if len(unique_values[col]) + len(values) <= UNIQUE_CAP:
                    unique_values[col].update(values.tolist())
                else:
                    unique_capped[col] = True
                    unique_values[col].clear()

        full_hash = pd.util.hash_pandas_object(chunk.fillna("<NA>"), index=False).to_numpy()
        duplicate_full_hashes.update(full_hash.tolist())
        key_cols = [col for col in LIKELY_KEY_COLUMNS if col in chunk.columns]
        key_hash = pd.util.hash_pandas_object(chunk[key_cols].fillna("<NA>"), index=False).to_numpy()
        duplicate_key_hashes.update(key_hash.tolist())
        if "ResultIdentifier" in chunk.columns:
            result_id_counter.update(chunk["ResultIdentifier"].dropna().astype(str).tolist())

        date = pd.to_datetime(chunk.get("ActivityStartDate"), errors="coerce")
        parsed_dates = date.dropna()
        date_parse_failures += int(date.isna().sum())
        if not parsed_dates.empty:
            cmin = parsed_dates.min()
            cmax = parsed_dates.max()
            date_min = cmin if date_min is None or cmin < date_min else date_min
            date_max = cmax if date_max is None or cmax > date_max else date_max
            years = parsed_dates.dt.year
            months = parsed_dates.dt.month
            rows_by_year.update(years.astype(int).tolist())
            rows_by_month.update(months.astype(int).tolist())
            rows_by_year_month.update(parsed_dates.dt.to_period("M").astype(str).tolist())
            rows_by_season.update([season_from_month(m) for m in months.tolist()])

        result_numeric = pd.to_numeric(chunk.get("ResultMeasureValue"), errors="coerce")
        result_raw = chunk.get("ResultMeasureValue", pd.Series(index=chunk.index, dtype="object")).astype("string")
        numeric_result_count += int(result_numeric.notna().sum())
        non_numeric_results += int(result_numeric.isna().sum())
        negative_numeric_count += int((result_numeric < 0).sum())
        result_measure_contains_lt += int(result_raw.str.contains("<", na=False, regex=False).sum())

        if "ResultDetectionConditionText" in chunk.columns:
            detection_text = normalize_text(chunk["ResultDetectionConditionText"])
            less_than_text_count += int(detection_text.str.contains("less|not detected|non-detect|below", case=False, na=False, regex=True).sum())
        if "ProviderName" in chunk.columns:
            provider_counter.update(normalize_text(chunk["ProviderName"]).dropna().astype(str).tolist())
        if "OrganizationIdentifier" in chunk.columns:
            organization_counter.update(normalize_text(chunk["OrganizationIdentifier"]).dropna().astype(str).tolist())

        site_ids = normalize_text(chunk.get("MonitoringLocationIdentifier", pd.Series(index=chunk.index, dtype="object")))
        site_types = site_ids.map(site_type_map).fillna("Unknown")
        site_type_record_counter.update(site_types.astype(str).tolist())

        lat = pd.to_numeric(chunk.get("ActivityLocation/LatitudeMeasure"), errors="coerce")
        lon = pd.to_numeric(chunk.get("ActivityLocation/LongitudeMeasure"), errors="coerce")
        lat = lat.fillna(site_ids.map(station_lat_map))
        lon = lon.fillna(site_ids.map(station_lon_map))
        invalid_latlon += int(((lat.notna()) & ((lat < -90) | (lat > 90))).sum())
        invalid_latlon += int(((lon.notna()) & ((lon < -180) | (lon > 180))).sum())
        florida_latlon_outside += int(((lat.notna()) & (lon.notna()) & ~((lat.between(24, 32)) & (lon.between(-88, -79)))).sum())

        char_col = normalize_text(chunk.get("CharacteristicName", pd.Series(index=chunk.index, dtype="object")))
        unit_col = normalize_unit(
            chunk.get("ResultMeasure/MeasureUnitCode", pd.Series(index=chunk.index, dtype="object")),
            chunk.get("CharacteristicName", pd.Series(index=chunk.index, dtype="object")),
        )
        provider_col = normalize_text(chunk.get("ProviderName", pd.Series(index=chunk.index, dtype="object"))).fillna("Missing provider")
        org_col = normalize_text(chunk.get("OrganizationIdentifier", pd.Series(index=chunk.index, dtype="object"))).fillna("Missing organization")
        non_detect_mask = pd.Series(False, index=chunk.index)
        if "ResultDetectionConditionText" in chunk.columns:
            non_detect_mask = non_detect_mask | chunk["ResultDetectionConditionText"].astype("string").str.contains(
                "less|not detected|non-detect|below", case=False, na=False, regex=True
            )
        non_detect_mask = non_detect_mask | result_raw.str.contains("<", na=False, regex=False)

        cframe = pd.DataFrame(
            {
                "characteristic": char_col,
                "unit": unit_col,
                "site": site_ids,
                "provider": provider_col,
                "organization": org_col,
                "year": date.dt.year,
                "date": date,
                "value": result_numeric,
                "non_detect": non_detect_mask,
                "site_type": site_types.astype(str),
            }
        )
        for char_name, group in cframe.groupby("characteristic", dropna=True):
            agg = char_aggs[str(char_name)]
            agg["record_count"] += int(len(group))
            agg["numeric_count"] += int(group["value"].notna().sum())
            agg["missing_result_count"] += int(group["value"].isna().sum())
            agg["negative_count"] += int((group["value"] < 0).sum())
            agg["non_detect_count"] += int(group["non_detect"].sum())
            agg["sites"].update(group["site"].dropna().astype(str).unique().tolist())
            agg["providers"].update(group["provider"].dropna().astype(str).tolist())
            agg["organizations"].update(group["organization"].dropna().astype(str).tolist())
            agg["units"].update(group["unit"].dropna().astype(str).tolist())
            agg["years"].update(group["year"].dropna().astype(int).unique().tolist())
            dvals = group["date"].dropna()
            if not dvals.empty:
                cmin = dvals.min()
                cmax = dvals.max()
                agg["min_date"] = cmin if agg["min_date"] is None or cmin < agg["min_date"] else agg["min_date"]
                agg["max_date"] = cmax if agg["max_date"] is None or cmax > agg["max_date"] else agg["max_date"]
            vals = group["value"].dropna()
            if not vals.empty:
                agg["min_value"] = min(float(vals.min()), agg["min_value"])
                agg["max_value"] = max(float(vals.max()), agg["max_value"])
        unit_summary_counter.update(zip(cframe["characteristic"].astype(str), cframe["unit"].astype(str)))
        site_type_char_counter.update(zip(cframe["site_type"].astype(str), cframe["characteristic"].astype(str)))

        sframe = pd.DataFrame(
            {
                "site": site_ids,
                "characteristic": char_col,
                "provider": provider_col,
                "organization": org_col,
                "year": date.dt.year,
                "month": date.dt.month,
                "date": date,
                "lat": lat,
                "lon": lon,
                "value": result_numeric,
            }
        ).dropna(subset=["site"])
        for site, group in sframe.groupby("site"):
            agg = site_aggs[str(site)]
            agg["record_count"] += int(len(group))
            agg["numeric_count"] += int(group["value"].notna().sum())
            agg["characteristics"].update(group["characteristic"].dropna().astype(str).unique().tolist())
            agg["providers"].update(group["provider"].dropna().astype(str).tolist())
            agg["organizations"].update(group["organization"].dropna().astype(str).tolist())
            agg["years"].update(group["year"].dropna().astype(int).unique().tolist())
            agg["months"].update(group["month"].dropna().astype(int).unique().tolist())
            dvals = group["date"].dropna()
            if not dvals.empty:
                cmin = dvals.min()
                cmax = dvals.max()
                agg["min_date"] = cmin if agg["min_date"] is None or cmin < agg["min_date"] else agg["min_date"]
                agg["max_date"] = cmax if agg["max_date"] is None or cmax > agg["max_date"] else agg["max_date"]
            agg["lat_values"].extend(group["lat"].dropna().head(3).astype(float).tolist())
            agg["lon_values"].extend(group["lon"].dropna().head(3).astype(float).tolist())

        clean = chunk[selected_cols].copy()
        clean.insert(0, "source_row_number", source_row_number)
        clean.insert(1, "source_row_key", [f"resultphyschem.csv:{row_num}" for row_num in source_row_number])
        clean["activity_date"] = date.dt.date
        clean["year"] = date.dt.year
        clean["month"] = date.dt.month
        clean["season"] = clean["month"].map(season_from_month)
        clean["result_value_numeric"] = result_numeric
        clean["normalized_result_unit"] = unit_col.values
        clean["non_detect_flag"] = non_detect_mask
        clean["site_type"] = site_types.values
        clean["station_latitude"] = lat.values
        clean["station_longitude"] = lon.values
        clean["county_code"] = site_ids.map(county_map).values
        clean["huc8"] = site_ids.map(huc_map).values
        clean.to_csv(cleaned_tmp_path, mode="w" if not cleaned_written else "a", index=False, header=not cleaned_written)
        cleaned_written = True
        processed_rows += len(clean)
        processed_numeric_rows += int(clean["result_value_numeric"].notna().sum())
        if "ResultIdentifier" in clean.columns:
            processed_result_id_counter.update(clean["ResultIdentifier"].dropna().astype(str).tolist())

    duplicate_full_rows = sum(count - 1 for count in duplicate_full_hashes.values() if count > 1)
    duplicate_key_rows = sum(count - 1 for count in duplicate_key_hashes.values() if count > 1)
    duplicate_result_id_rows = sum(count - 1 for count in result_id_counter.values() if count > 1)
    processed_duplicate_result_id_rows = sum(count - 1 for count in processed_result_id_counter.values() if count > 1)
    processed_duplicate_row_key_rows = 0
    if cleaned_tmp_path.exists():
        cleaned_tmp_path.replace(cleaned_path)

    profile = {
        "total_rows": total_rows,
        "total_columns": len(header),
        "header": header,
        "result_size_mb": bytes_to_mb(RESULT_PATH.stat().st_size),
        "station_size_mb": bytes_to_mb(STATION_PATH.stat().st_size),
        "sample_memory_mb": round(sample_memory_mb, 2),
        "estimated_memory_mb": round(estimated_memory_mb, 2) if not pd.isna(estimated_memory_mb) else np.nan,
        "date_min": date_min,
        "date_max": date_max,
        "date_parse_failures": date_parse_failures,
        "numeric_result_count": numeric_result_count,
        "non_numeric_results": non_numeric_results,
        "negative_numeric_count": negative_numeric_count,
        "result_measure_contains_lt": result_measure_contains_lt,
        "less_than_text_count": less_than_text_count,
        "invalid_latlon": invalid_latlon,
        "florida_latlon_outside": florida_latlon_outside,
        "duplicate_full_rows": duplicate_full_rows,
        "duplicate_key_rows": duplicate_key_rows,
        "duplicate_result_id_rows": duplicate_result_id_rows,
        "processed_rows": processed_rows,
        "processed_numeric_rows": processed_numeric_rows,
        "processed_duplicate_result_id_rows": processed_duplicate_result_id_rows,
        "processed_duplicate_row_key_rows": processed_duplicate_row_key_rows,
        "processed_rows_removed": total_rows - processed_rows,
        "processed_rows_removed_reason": "None; row-preserving selected-column output.",
        "missing_counts": missing_counts,
        "inferred_dtypes": inferred_dtypes,
        "unique_values": unique_values,
        "unique_capped": unique_capped,
        "examples": examples,
        "provider_counter": provider_counter,
        "organization_counter": organization_counter,
        "site_type_record_counter": site_type_record_counter,
        "rows_by_year": rows_by_year,
        "rows_by_month": rows_by_month,
        "rows_by_year_month": rows_by_year_month,
        "rows_by_season": rows_by_season,
        "char_aggs": char_aggs,
        "unit_summary_counter": unit_summary_counter,
        "site_type_char_counter": site_type_char_counter,
        "site_aggs": site_aggs,
        "station": station,
        "cleaned_path": cleaned_path,
    }
    return profile


def estimate_row_count_fast() -> int:
    # The full chunked pass calculates the final row count. This estimate is
    # only for the memory projection printed before the pass has completed.
    with RESULT_PATH.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        line_count = sum(1 for _ in handle)
    return max(line_count - 1, 0)


def write_outputs(profile: dict[str, object]) -> None:
    total_rows = int(profile["total_rows"])
    header = profile["header"]
    missing_counts: pd.Series = profile["missing_counts"]

    repo_summary = pd.DataFrame(
        [
            {
                "file_role": "raw WQP physical/chemical results",
                "path": str(RESULT_PATH.relative_to(BASE_DIR)),
                "size_mb": profile["result_size_mb"],
                "rows": total_rows,
                "columns": profile["total_columns"],
                "sample_10000_memory_mb": profile["sample_memory_mb"],
                "estimated_full_pandas_memory_mb": profile["estimated_memory_mb"],
                "processing_recommendation": (
                    "Chunked pandas is safest for reproducible EDA; full pandas may work on high-memory machines."
                    if profile["estimated_memory_mb"] >= 1_500
                    else "Pandas is sufficient; chunking still used for reproducibility."
                ),
            },
            {
                "file_role": "raw WQP station metadata",
                "path": str(STATION_PATH.relative_to(BASE_DIR)),
                "size_mb": profile["station_size_mb"],
                "rows": len(profile["station"]),
                "columns": profile["station"].shape[1],
                "sample_10000_memory_mb": "",
                "estimated_full_pandas_memory_mb": "",
                "processing_recommendation": "Small enough for pandas.",
            },
        ]
    )
    repo_summary.to_csv(EDA_DIR / "wqp_repository_data_inventory.csv", index=False)

    dictionary_rows = []
    for col in header:
        missing = int(missing_counts[col])
        missing_pct = missing / total_rows * 100 if total_rows else np.nan
        if profile["unique_capped"][col]:
            unique_count = f">={UNIQUE_CAP:,}"
        else:
            unique_count = str(len(profile["unique_values"][col]))
        useful, reason = useful_column_reason(col, missing_pct, unique_count)
        dictionary_rows.append(
            {
                "column_name": col,
                "pandas_dtype_inferred": "; ".join(sorted(profile["inferred_dtypes"][col])),
                "example_values": " | ".join(profile["examples"][col]),
                "missing_count": missing,
                "missing_pct": round(missing_pct, 3),
                "unique_values": unique_count,
                "analysis_role": classify_column(col),
                "likely_useful_for_analysis": useful,
                "notes": reason,
            }
        )
    data_dictionary = pd.DataFrame(dictionary_rows)
    data_dictionary.to_csv(EDA_DIR / "wqp_data_dictionary.csv", index=False)

    missingness = data_dictionary[["column_name", "missing_count", "missing_pct", "analysis_role"]].sort_values(
        "missing_pct", ascending=False
    )
    missingness.to_csv(EDA_DIR / "wqp_missingness_summary.csv", index=False)

    char_rows = []
    for char_name, agg in profile["char_aggs"].items():
        min_val = agg["min_value"] if agg["min_value"] != math.inf else np.nan
        max_val = agg["max_value"] if agg["max_value"] != -math.inf else np.nan
        char_rows.append(
            {
                "CharacteristicName": char_name,
                "derived_characteristic_group": classify_characteristic_group(char_name),
                "record_count": agg["record_count"],
                "record_pct": agg["record_count"] / total_rows * 100 if total_rows else np.nan,
                "numeric_count": agg["numeric_count"],
                "numeric_pct_within_characteristic": agg["numeric_count"] / agg["record_count"] * 100
                if agg["record_count"]
                else np.nan,
                "missing_result_count": agg["missing_result_count"],
                "negative_count": agg["negative_count"],
                "non_detect_count": agg["non_detect_count"],
                "site_count": len(agg["sites"]),
                "provider_count": len(agg["providers"]),
                "organization_count": len(agg["organizations"]),
                "unit_count": len(agg["units"]),
                "dominant_unit": agg["units"].most_common(1)[0][0] if agg["units"] else "",
                "dominant_unit_count": agg["units"].most_common(1)[0][1] if agg["units"] else 0,
                "dominant_unit_pct": (
                    agg["units"].most_common(1)[0][1] / agg["record_count"] * 100
                    if agg["units"] and agg["record_count"]
                    else np.nan
                ),
                "top_units": safe_value_counts(agg["units"], 6),
                "active_years": len(agg["years"]),
                "all_baseline_years_present": set(agg["years"]).issuperset(EXPECTED_BASELINE_YEARS),
                "three_year_continuity_candidate": len(agg["years"]) >= MIN_ACTIVE_YEARS_FOR_CONTINUITY,
                "first_date": agg["min_date"].date().isoformat() if agg["min_date"] is not None else "",
                "last_date": agg["max_date"].date().isoformat() if agg["max_date"] is not None else "",
                "min_numeric_value": min_val,
                "max_numeric_value": max_val,
                "modeling_candidate": is_modeling_candidate(agg),
                "candidate_reason": candidate_reason(agg),
            }
        )
    characteristic_summary = pd.DataFrame(char_rows).sort_values("record_count", ascending=False)
    characteristic_summary.to_csv(EDA_DIR / "wqp_characteristic_summary.csv", index=False)

    unit_rows = [
        {"CharacteristicName": char, "unit": unit, "record_count": count}
        for (char, unit), count in profile["unit_summary_counter"].items()
    ]
    unit_summary = pd.DataFrame(unit_rows).sort_values(["CharacteristicName", "record_count"], ascending=[True, False])
    unit_summary.to_csv(EDA_DIR / "wqp_unit_summary.csv", index=False)

    site_rows = []
    station = profile["station"].set_index("MonitoringLocationIdentifier")
    for site, agg in profile["site_aggs"].items():
        station_row = station.loc[site] if site in station.index else None
        lat = np.nanmedian(agg["lat_values"]) if agg["lat_values"] else np.nan
        lon = np.nanmedian(agg["lon_values"]) if agg["lon_values"] else np.nan
        low_record_count_flag = agg["record_count"] < LOW_RECORD_COUNT_THRESHOLD
        low_characteristic_count_flag = len(agg["characteristics"]) < LOW_CHARACTERISTIC_COUNT_THRESHOLD
        single_year_coverage_flag = len(agg["years"]) <= 1
        limited_month_coverage_flag = len(agg["months"]) < LIMITED_MONTH_COVERAGE_THRESHOLD
        missing_coordinates_flag = pd.isna(lat) or pd.isna(lon)
        no_numeric_results_flag = agg["numeric_count"] == 0
        modeling_sparse_flag = low_record_count_flag or low_characteristic_count_flag or no_numeric_results_flag
        site_rows.append(
            {
                "MonitoringLocationIdentifier": site,
                "MonitoringLocationName": station_row.get("MonitoringLocationName", "") if station_row is not None else "",
                "site_type": station_row.get("MonitoringLocationTypeName", "Unknown") if station_row is not None else "Unknown",
                "county_code": station_row.get("CountyCode", "") if station_row is not None else "",
                "huc8": station_row.get("HUCEightDigitCode", "") if station_row is not None else "",
                "latitude": lat,
                "longitude": lon,
                "record_count": agg["record_count"],
                "numeric_count": agg["numeric_count"],
                "characteristic_count": len(agg["characteristics"]),
                "provider_count": len(agg["providers"]),
                "top_providers": safe_value_counts(agg["providers"], 3),
                "organization_count": len(agg["organizations"]),
                "active_years": len(agg["years"]),
                "active_months": len(agg["months"]),
                "first_date": agg["min_date"].date().isoformat() if agg["min_date"] is not None else "",
                "last_date": agg["max_date"].date().isoformat() if agg["max_date"] is not None else "",
                "low_record_count_flag": low_record_count_flag,
                "low_characteristic_count_flag": low_characteristic_count_flag,
                "single_year_coverage_flag": single_year_coverage_flag,
                "limited_month_coverage_flag": limited_month_coverage_flag,
                "missing_coordinates_flag": missing_coordinates_flag,
                "no_numeric_results_flag": no_numeric_results_flag,
                "modeling_sparse_flag": modeling_sparse_flag,
            }
        )
    site_summary = pd.DataFrame(site_rows).sort_values("record_count", ascending=False)
    site_summary.to_csv(EDA_DIR / "wqp_site_summary.csv", index=False)

    temporal_rows = []
    for year, count in sorted(profile["rows_by_year"].items()):
        temporal_rows.append({"period_type": "year", "period": str(year), "record_count": count})
    for ym, count in sorted(profile["rows_by_year_month"].items()):
        temporal_rows.append({"period_type": "year_month", "period": str(ym), "record_count": count})
    for month, count in sorted(profile["rows_by_month"].items()):
        temporal_rows.append({"period_type": "calendar_month", "period": str(month), "record_count": count})
    for season, count in profile["rows_by_season"].most_common():
        temporal_rows.append({"period_type": "season", "period": str(season), "record_count": count})
    temporal_summary = pd.DataFrame(temporal_rows)
    temporal_summary.to_csv(EDA_DIR / "wqp_temporal_summary.csv", index=False)

    site_type_char = pd.DataFrame(
        [
            {"site_type": site_type, "CharacteristicName": char, "record_count": count}
            for (site_type, char), count in profile["site_type_char_counter"].items()
        ]
    )
    site_type_char.to_csv(EDA_DIR / "wqp_site_type_characteristic_summary.csv", index=False)

    common_rows = [
        summarize_common_variable(group, characteristic_summary)
        for group in COMMON_VARIABLE_PATTERNS
    ]
    common_summary = pd.DataFrame(common_rows)
    common_summary.to_csv(EDA_DIR / "wqp_common_variable_presence.csv", index=False)

    candidate_vars = characteristic_summary.loc[characteristic_summary["modeling_candidate"]].copy()
    candidate_vars.to_csv(EDA_DIR / "wqp_candidate_modeling_variables.csv", index=False)

    quality_issues = pd.DataFrame(
        [
            quality_issue("Duplicate full CSV rows", profile["duplicate_full_rows"], "Potential exact duplicated records."),
            quality_issue("Duplicate likely result key rows", profile["duplicate_key_rows"], "Likely duplicated analytical result records."),
            quality_issue("Duplicate ResultIdentifier rows", profile["duplicate_result_id_rows"], "ResultIdentifier should usually be unique."),
            quality_issue("Date parse failures", profile["date_parse_failures"], "Rows where ActivityStartDate could not be parsed."),
            quality_issue("Non-numeric ResultMeasureValue rows", profile["non_numeric_results"], "Rows that cannot be used directly in numeric modeling."),
            quality_issue("Negative numeric result rows", profile["negative_numeric_count"], "May be valid for a few parameters but suspicious for concentrations/counts."),
            quality_issue("ResultMeasureValue contains '<'", profile["result_measure_contains_lt"], "Censored values embedded in result text."),
            quality_issue("Detection/non-detect text rows", profile["less_than_text_count"], "Rows flagged as less-than, below detection, or non-detect."),
            quality_issue("Invalid latitude/longitude rows", profile["invalid_latlon"], "Coordinates outside valid latitude/longitude ranges."),
            quality_issue("Coordinates outside broad Florida bounds", profile["florida_latlon_outside"], "Rows with site coordinates outside 24-32 N and -88 to -79 W."),
        ]
    )
    quality_issues.to_csv(EDA_DIR / "wqp_data_quality_issues.csv", index=False)

    summarize_counter(profile["provider_counter"], total_rows, "ProviderName").to_csv(
        EDA_DIR / "wqp_provider_summary.csv", index=False
    )
    summarize_counter(profile["organization_counter"], total_rows, "OrganizationIdentifier").to_csv(
        EDA_DIR / "wqp_organization_summary.csv", index=False
    )
    summarize_counter(profile["site_type_record_counter"], total_rows, "site_type").to_csv(
        EDA_DIR / "wqp_site_type_summary.csv", index=False
    )

    matrix_validation = build_site_characteristic_matrix(profile, candidate_vars)
    write_numeric_range_screening(candidate_vars)
    write_processed_validation_summary(profile, candidate_vars, matrix_validation)
    technique_df = build_technique_matrix(profile, characteristic_summary, site_summary, candidate_vars)
    technique_df.to_csv(EDA_DIR / "wqp_technique_feasibility_matrix.csv", index=False)
    make_figures(profile, characteristic_summary, site_summary, missingness, unit_summary, site_type_char)


def dominant_unit_pct(agg: dict[str, object]) -> float:
    if not agg["units"] or not agg["record_count"]:
        return 0.0
    return agg["units"].most_common(1)[0][1] / agg["record_count"] * 100


def is_modeling_candidate(agg: dict[str, object]) -> bool:
    return (
        agg["record_count"] >= MIN_CANDIDATE_ROWS
        and agg["numeric_count"] >= MIN_CANDIDATE_ROWS
        and len(agg["sites"]) >= MIN_CANDIDATE_SITES
        and dominant_unit_pct(agg) >= 70
    )


def candidate_reason(agg: dict[str, object]) -> str:
    problems = []
    if agg["record_count"] < MIN_CANDIDATE_ROWS:
        problems.append("too few rows")
    if agg["numeric_count"] < MIN_CANDIDATE_ROWS:
        problems.append("too few numeric rows")
    if len(agg["sites"]) < MIN_CANDIDATE_SITES:
        problems.append("too few sites")
    if dominant_unit_pct(agg) < 70:
        problems.append("no dominant unit; mixed units need conversion or filtering")
    elif len(agg["units"]) > 1:
        problems.append("candidate after filtering/reviewing dominant unit")
    if not set(agg["years"]).issuperset(EXPECTED_BASELINE_YEARS):
        problems.append("not sampled in all verified baseline years; review before temporal use")
    return "Strong 2021-2023 modeling-screening candidate" if not problems else "; ".join(problems)


def quality_issue(name: str, count: int, notes: str) -> dict[str, object]:
    return {"issue": name, "affected_rows_or_count": int(count), "notes": notes}


def build_site_characteristic_matrix(profile: dict[str, object], candidate_vars: pd.DataFrame) -> pd.DataFrame:
    all_candidates = candidate_vars["CharacteristicName"].astype(str).tolist()
    matrix_candidates = all_candidates[:20]
    validation_rows = []
    if not matrix_candidates:
        pd.DataFrame(columns=["MonitoringLocationIdentifier"]).to_csv(
            PROCESSED_DIR / "florida_wqp_site_characteristic_matrix.csv", index=False
        )
        pd.DataFrame(columns=[
            "CharacteristicName",
            "selected_dominant_unit",
            "usable_numeric_row_count",
            "site_count",
            "active_years",
            "included_in_matrix",
            "exclusion_reason",
        ]).to_csv(EDA_DIR / "wqp_candidate_matrix_validation.csv", index=False)
        return pd.DataFrame()

    unit_filters = {}
    metadata_rows = []
    for char in matrix_candidates:
        units = profile["char_aggs"][char]["units"]
        if not units:
            continue
        dominant_unit, dominant_count = units.most_common(1)[0]
        dominant_pct = dominant_count / profile["char_aggs"][char]["record_count"] * 100
        if dominant_pct >= 70:
            unit_filters[char] = dominant_unit
            metadata_rows.append(
                {
                    "CharacteristicName": char,
                    "matrix_unit_filter": dominant_unit,
                    "dominant_unit_pct": dominant_pct,
                    "notes": "Site matrix uses only the dominant unit for this characteristic.",
                }
            )
    pd.DataFrame(metadata_rows).to_csv(PROCESSED_DIR / "florida_wqp_site_characteristic_matrix_metadata.csv", index=False)
    matrix_candidates = list(unit_filters)
    usable_numeric_counter = Counter()
    usable_site_sets = defaultdict(set)
    if not matrix_candidates:
        pd.DataFrame(columns=["MonitoringLocationIdentifier"]).to_csv(
            PROCESSED_DIR / "florida_wqp_site_characteristic_matrix.csv", index=False
        )
        return pd.DataFrame()

    pieces = []
    usecols = [
        "MonitoringLocationIdentifier",
        "CharacteristicName",
        "ResultMeasureValue",
        "ResultMeasure/MeasureUnitCode",
        "ActivityStartDate",
    ]
    for chunk in pd.read_csv(RESULT_PATH, chunksize=CHUNKSIZE, usecols=usecols, low_memory=False):
        chunk = chunk[chunk["CharacteristicName"].astype(str).isin(matrix_candidates)].copy()
        if chunk.empty:
            continue
        normalized_units = normalize_unit(chunk["ResultMeasure/MeasureUnitCode"], chunk["CharacteristicName"])
        expected_unit = chunk["CharacteristicName"].astype(str).map(unit_filters)
        unit_mask = normalized_units.astype(str).eq(expected_unit.astype(str))
        chunk = chunk[unit_mask].copy()
        if chunk.empty:
            continue
        chunk["result_value_numeric"] = pd.to_numeric(chunk["ResultMeasureValue"], errors="coerce")
        chunk = chunk.dropna(subset=["MonitoringLocationIdentifier", "CharacteristicName", "result_value_numeric"])
        if not chunk.empty:
            for char, group in chunk.groupby("CharacteristicName"):
                usable_numeric_counter[str(char)] += int(len(group))
                usable_site_sets[str(char)].update(group["MonitoringLocationIdentifier"].dropna().astype(str).unique().tolist())
            pieces.append(chunk[["MonitoringLocationIdentifier", "CharacteristicName", "result_value_numeric"]])
    if not pieces:
        pd.DataFrame(columns=["MonitoringLocationIdentifier"]).to_csv(
            PROCESSED_DIR / "florida_wqp_site_characteristic_matrix.csv", index=False
        )
        matrix = pd.DataFrame(columns=["MonitoringLocationIdentifier"])
    else:
        long = pd.concat(pieces, ignore_index=True)
        counts = long.groupby(["MonitoringLocationIdentifier", "CharacteristicName"]).size().rename("n").reset_index()
        medians = long.groupby(["MonitoringLocationIdentifier", "CharacteristicName"])["result_value_numeric"].median().rename("median").reset_index()
        merged = medians.merge(counts, on=["MonitoringLocationIdentifier", "CharacteristicName"])
        merged.loc[merged["n"] < MATRIX_MIN_OBS_PER_SITE_CHAR, "median"] = np.nan
        matrix = merged.pivot(index="MonitoringLocationIdentifier", columns="CharacteristicName", values="median").reset_index()
        matrix.to_csv(PROCESSED_DIR / "florida_wqp_site_characteristic_matrix.csv", index=False)

    matrix_columns = set(matrix.columns) - {"MonitoringLocationIdentifier"}
    for char in all_candidates:
        agg = profile["char_aggs"][char]
        selected_unit = unit_filters.get(char, "")
        if char in matrix_columns:
            reason = ""
        elif char not in unit_filters:
            reason = "Not selected for matrix top-20 candidate subset or no dominant unit filter."
        elif usable_numeric_counter[char] == 0:
            reason = "No usable numeric rows after unit filtering."
        else:
            reason = "No site-characteristic pairs met the minimum observation threshold."
        validation_rows.append(
            {
                "CharacteristicName": char,
                "selected_dominant_unit": selected_unit,
                "usable_numeric_row_count": int(usable_numeric_counter[char]),
                "site_count": len(usable_site_sets[char]) if char in usable_site_sets else 0,
                "active_years": len(agg["years"]),
                "included_in_matrix": char in matrix_columns,
                "exclusion_reason": reason,
            }
        )
    validation = pd.DataFrame(validation_rows)
    validation.to_csv(EDA_DIR / "wqp_candidate_matrix_validation.csv", index=False)
    return validation


def write_numeric_range_screening(candidate_vars: pd.DataFrame) -> pd.DataFrame:
    candidate_units = {
        str(row["CharacteristicName"]): str(row["dominant_unit"])
        for _, row in candidate_vars.iterrows()
    }
    candidates = set(candidate_units)
    rows = []
    usecols = [
        "MonitoringLocationIdentifier",
        "CharacteristicName",
        "ResultMeasureValue",
        "ResultMeasure/MeasureUnitCode",
    ]
    for chunk in pd.read_csv(RESULT_PATH, chunksize=CHUNKSIZE, usecols=usecols, low_memory=False):
        chunk = chunk[chunk["CharacteristicName"].astype(str).isin(candidates)].copy()
        if chunk.empty:
            continue
        chunk["normalized_result_unit"] = normalize_unit(chunk["ResultMeasure/MeasureUnitCode"], chunk["CharacteristicName"])
        expected_unit = chunk["CharacteristicName"].astype(str).map(candidate_units)
        chunk = chunk[chunk["normalized_result_unit"].astype(str).eq(expected_unit.astype(str))].copy()
        if chunk.empty:
            continue
        chunk["result_value_numeric"] = pd.to_numeric(chunk["ResultMeasureValue"], errors="coerce")
        chunk = chunk.dropna(subset=["result_value_numeric"])
        if not chunk.empty:
            rows.append(
                chunk[
                    [
                        "MonitoringLocationIdentifier",
                        "CharacteristicName",
                        "normalized_result_unit",
                        "result_value_numeric",
                    ]
                ]
            )

    if not rows:
        out = pd.DataFrame(
            columns=[
                "CharacteristicName",
                "unit",
                "count",
                "site_count",
                "min",
                "p01",
                "p05",
                "median",
                "mean",
                "p95",
                "p99",
                "max",
                "negative_count",
                "zero_count",
                "extreme_high_count_iqr",
                "extreme_high_threshold_iqr",
            ]
        )
        out.to_csv(EDA_DIR / "wqp_numeric_range_screening.csv", index=False)
        return out

    data = pd.concat(rows, ignore_index=True)
    summary_rows = []
    for (char, unit), group in data.groupby(["CharacteristicName", "normalized_result_unit"], dropna=False):
        values = group["result_value_numeric"].dropna()
        if values.empty:
            continue
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        high_threshold = q3 + 1.5 * iqr
        summary_rows.append(
            {
                "CharacteristicName": char,
                "unit": unit,
                "count": int(values.count()),
                "site_count": int(group["MonitoringLocationIdentifier"].nunique()),
                "min": values.min(),
                "p01": values.quantile(0.01),
                "p05": values.quantile(0.05),
                "median": values.median(),
                "mean": values.mean(),
                "p95": values.quantile(0.95),
                "p99": values.quantile(0.99),
                "max": values.max(),
                "negative_count": int((values < 0).sum()),
                "zero_count": int((values == 0).sum()),
                "extreme_high_count_iqr": int((values > high_threshold).sum()),
                "extreme_high_threshold_iqr": high_threshold,
            }
        )
    out = pd.DataFrame(summary_rows).sort_values(["count", "CharacteristicName"], ascending=[False, True])
    out.to_csv(EDA_DIR / "wqp_numeric_range_screening.csv", index=False)
    return out


def write_processed_validation_summary(
    profile: dict[str, object],
    candidate_vars: pd.DataFrame,
    matrix_validation: pd.DataFrame,
) -> pd.DataFrame:
    matrix_path = PROCESSED_DIR / "florida_wqp_site_characteristic_matrix.csv"
    if matrix_path.exists() and matrix_path.stat().st_size:
        try:
            matrix = pd.read_csv(matrix_path)
        except pd.errors.EmptyDataError:
            matrix = pd.DataFrame()
    else:
        matrix = pd.DataFrame()

    candidate_names = set(candidate_vars["CharacteristicName"].astype(str))
    matrix_columns = set(matrix.columns) - {"MonitoringLocationIdentifier"}
    ph_candidate = "pH" in candidate_names
    ph_in_matrix = "pH" in matrix_columns
    active_years = sorted(profile["rows_by_year"].keys())
    active_year_set = set(active_years)
    expected_years_present = EXPECTED_BASELINE_YEARS.issubset(active_year_set)
    unexpected_years = sorted(active_year_set - EXPECTED_BASELINE_YEARS)
    missing_expected_years = sorted(EXPECTED_BASELINE_YEARS - active_year_set)
    validation_rows = [
        {"check": "raw_row_count", "value": profile["total_rows"], "notes": "Parsed rows in raw resultphyschem.csv."},
        {"check": "processed_numeric_long_row_count", "value": profile["processed_rows"], "notes": "Rows written to row-preserving processed long CSV."},
        {"check": "processed_row_count_difference", "value": profile["processed_rows"] - profile["total_rows"], "notes": "Should be 0 for the row-preserving output."},
        {"check": "rows_with_parsed_numeric_values", "value": profile["processed_numeric_rows"], "notes": "Rows with numeric ResultMeasureValue after parsing."},
        {"check": "non_numeric_or_missing_result_rows", "value": profile["non_numeric_results"], "notes": "Rows with missing or non-numeric ResultMeasureValue."},
        {"check": "raw_duplicate_ResultIdentifier_rows", "value": profile["duplicate_result_id_rows"], "notes": "Duplicate ResultIdentifier rows in raw data."},
        {"check": "processed_duplicate_ResultIdentifier_rows", "value": profile["processed_duplicate_result_id_rows"], "notes": "Duplicate ResultIdentifier rows in processed long data."},
        {"check": "processed_duplicate_constructed_row_key_rows", "value": profile["processed_duplicate_row_key_rows"], "notes": "Constructed source_row_key is based on raw source row order."},
        {"check": "processed_rows_removed", "value": profile["processed_rows_removed"], "notes": profile["processed_rows_removed_reason"]},
        {"check": "pH_in_candidate_metadata", "value": ph_candidate, "notes": "Whether pH is selected as a candidate characteristic."},
        {"check": "pH_in_site_matrix", "value": ph_in_matrix, "notes": "Whether pH appears as a site-characteristic matrix column."},
        {"check": "site_matrix_rows", "value": len(matrix), "notes": "Number of site rows in matrix."},
        {"check": "site_matrix_columns", "value": len(matrix.columns), "notes": "Includes MonitoringLocationIdentifier when present."},
        {"check": "candidate_variable_count", "value": len(candidate_vars), "notes": "Number of candidate variables selected from EDA."},
        {"check": "active_year_min", "value": min(active_years) if active_years else "", "notes": "Minimum parsed activity year."},
        {"check": "active_year_max", "value": max(active_years) if active_years else "", "notes": "Maximum parsed activity year."},
        {"check": "number_of_active_years", "value": len(active_years), "notes": "Count of active years in current export."},
        {"check": "expected_2021_2023_years_present", "value": expected_years_present, "notes": f"Missing expected years: {missing_expected_years}."},
        {"check": "unexpected_activity_years", "value": "; ".join(map(str, unexpected_years)), "notes": "Unexpected years outside the verified 2021-2023 baseline."},
    ]
    validation = pd.DataFrame(validation_rows)
    validation.to_csv(EDA_DIR / "wqp_processed_validation_summary.csv", index=False)
    print(validation.to_string(index=False))

    if profile["processed_rows"] != profile["total_rows"]:
        raise RuntimeError("Processed long file is not row-preserving.")
    if profile["processed_duplicate_result_id_rows"] != profile["duplicate_result_id_rows"]:
        raise RuntimeError("Processed long file ResultIdentifier duplicate count differs from raw.")
    if ph_candidate and not ph_in_matrix:
        raise RuntimeError("pH is a candidate but is missing from the site-characteristic matrix.")
    if not expected_years_present or unexpected_years:
        raise RuntimeError("ActivityStartDate coverage does not match the verified 2021-2023 baseline.")
    return validation


def make_figures(
    profile: dict[str, object],
    characteristic_summary: pd.DataFrame,
    site_summary: pd.DataFrame,
    missingness: pd.DataFrame,
    unit_summary: pd.DataFrame,
    site_type_char: pd.DataFrame,
) -> None:
    year_df = pd.DataFrame(
        [{"year": year, "record_count": count} for year, count in sorted(profile["rows_by_year"].items())]
    )
    save_bar(year_df, "year", "record_count", FIG_DIR / "records_by_year.png", "Florida WQP Records by Year", "Year", "Records")

    month_df = pd.DataFrame(
        [{"month": month, "record_count": count} for month, count in sorted(profile["rows_by_month"].items())]
    )
    save_bar(month_df, "month", "record_count", FIG_DIR / "records_by_month.png", "Florida WQP Records by Calendar Month", "Month", "Records")

    top_char = characteristic_summary.head(20).sort_values("record_count", ascending=True)
    save_horizontal_bar(
        top_char,
        "CharacteristicName",
        "record_count",
        FIG_DIR / "top_20_characteristics.png",
        "Top 20 Characteristics by Record Count",
        "Records",
        "Characteristic",
        width=30,
    )

    site_type_df = pd.DataFrame(
        [{"site_type": key, "record_count": count} for key, count in profile["site_type_record_counter"].most_common(15)]
    ).sort_values("record_count", ascending=True)
    save_horizontal_bar(
        site_type_df,
        "site_type",
        "record_count",
        FIG_DIR / "top_site_types.png",
        "Records by Site Type",
        "Records",
        "Site type",
        width=24,
    )

    miss = missingness.sort_values("missing_pct", ascending=False).head(35).sort_values("missing_pct")
    miss = miss.assign(column_name=miss["column_name"].map(lambda item: wrap_label(item, 34)))
    plt.figure(figsize=(12, 11))
    plt.barh(miss["column_name"], miss["missing_pct"], color="#F58518")
    plt.title("Highest Missingness Columns")
    plt.xlabel("Missing percent")
    plt.ylabel("Column")
    plt.xlim(0, 100)
    plt.grid(axis="x", alpha=0.2)
    save_current_figure(FIG_DIR / "missingness_bar.png")

    provider_df = pd.DataFrame(
        [{"ProviderName": key, "record_count": count} for key, count in profile["provider_counter"].most_common(10)]
    )
    save_bar(provider_df, "ProviderName", "record_count", FIG_DIR / "provider_contribution.png", "Provider Contribution", "Provider", "Records", rotate=20)

    map_df = site_summary.dropna(subset=["latitude", "longitude"]).copy()
    map_df = map_df[(map_df["latitude"].between(24, 32)) & (map_df["longitude"].between(-88, -79))]
    if not map_df.empty:
        plt.figure(figsize=(8.5, 9.5))
        size = np.clip(np.log1p(map_df["record_count"]) * 8, 8, 80)
        plt.scatter(map_df["longitude"], map_df["latitude"], s=size, alpha=0.45, c="#4C78A8", edgecolors="none")
        plt.title("Monitoring Site Geographic Coverage")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.grid(alpha=0.15)
        save_current_figure(FIG_DIR / "monitoring_sites_map_scatter.png")

    common_for_box = characteristic_summary[
        characteristic_summary["CharacteristicName"].astype(str).str.lower().isin(
            ["ph", "temperature, water", "specific conductance", "dissolved oxygen (do)", "turbidity"]
        )
    ]["CharacteristicName"].astype(str).tolist()
    make_distribution_plot(common_for_box)

    top_unit_chars = characteristic_summary.head(15)["CharacteristicName"].astype(str).tolist()
    unit_sub = unit_summary[unit_summary["CharacteristicName"].astype(str).isin(top_unit_chars)]
    if not unit_sub.empty:
        pivot = unit_sub.pivot_table(index="CharacteristicName", columns="unit", values="record_count", aggfunc="sum", fill_value=0)
        pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]
        plt.figure(figsize=(16, 9))
        if sns is not None:
            ax = sns.heatmap(np.log1p(pivot), cmap="viridis", cbar_kws={"label": "log(1 + records)"})
            ax.set_xticklabels([wrap_label(label.get_text(), 12) for label in ax.get_xticklabels()], rotation=45, ha="right", fontsize=8)
            ax.set_yticklabels([wrap_label(label.get_text(), 28) for label in ax.get_yticklabels()], rotation=0, fontsize=8)
        else:
            plt.imshow(np.log1p(pivot), aspect="auto", cmap="viridis")
            plt.yticks(range(len(pivot.index)), pivot.index)
            plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
            plt.colorbar(label="log(1 + records)")
        plt.title("Unit Consistency for Top Characteristics")
        plt.xlabel("Unit")
        plt.ylabel("Characteristic")
        save_current_figure(FIG_DIR / "unit_consistency_heatmap.png")

    if not site_type_char.empty:
        top_chars = characteristic_summary.head(20)["CharacteristicName"].astype(str)
        top_site_types = site_type_char.groupby("site_type")["record_count"].sum().sort_values(ascending=False).head(12).index
        heat = site_type_char[
            site_type_char["CharacteristicName"].astype(str).isin(top_chars)
            & site_type_char["site_type"].astype(str).isin(top_site_types)
        ].pivot_table(index="site_type", columns="CharacteristicName", values="record_count", aggfunc="sum", fill_value=0)
        if not heat.empty:
            plt.figure(figsize=(18, 8))
            if sns is not None:
                ax = sns.heatmap(np.log1p(heat), cmap="mako", cbar_kws={"label": "log(1 + records)"})
                ax.set_xticklabels([wrap_label(label.get_text(), 16) for label in ax.get_xticklabels()], rotation=45, ha="right", fontsize=8)
                ax.set_yticklabels([wrap_label(label.get_text(), 24) for label in ax.get_yticklabels()], rotation=0, fontsize=8)
            else:
                plt.imshow(np.log1p(heat), aspect="auto", cmap="mako")
                plt.yticks(range(len(heat.index)), heat.index)
                plt.xticks(range(len(heat.columns)), heat.columns, rotation=45, ha="right")
                plt.colorbar(label="log(1 + records)")
            plt.title("Site Type vs Characteristic Availability")
            plt.xlabel("Characteristic")
            plt.ylabel("Site type")
            save_current_figure(FIG_DIR / "site_type_characteristic_heatmap.png")

    make_correlation_heatmap()


def make_distribution_plot(characteristics: list[str]) -> None:
    if not characteristics:
        return
    rows = []
    usecols = ["CharacteristicName", "ResultMeasureValue", "ResultMeasure/MeasureUnitCode"]
    for chunk in pd.read_csv(RESULT_PATH, chunksize=CHUNKSIZE, usecols=usecols, low_memory=False):
        sub = chunk[chunk["CharacteristicName"].astype(str).isin(characteristics)].copy()
        if sub.empty:
            continue
        sub["result_value_numeric"] = pd.to_numeric(sub["ResultMeasureValue"], errors="coerce")
        sub = sub.dropna(subset=["result_value_numeric"])
        for char, group in sub.groupby("CharacteristicName"):
            if len(group) > 3_000:
                group = group.sample(3_000, random_state=42)
            rows.append(group[["CharacteristicName", "result_value_numeric"]])
    if not rows:
        return
    data = pd.concat(rows, ignore_index=True)
    data["CharacteristicName"] = data["CharacteristicName"].map(lambda item: wrap_label(item, 18))
    plt.figure(figsize=(12, 7.5))
    if sns is not None:
        sns.boxplot(data=data, x="CharacteristicName", y="result_value_numeric", showfliers=False, color="#72B7B2")
    else:
        labels = []
        values = []
        for char, group in data.groupby("CharacteristicName"):
            labels.append(char)
            values.append(group["result_value_numeric"].to_numpy())
        plt.boxplot(values, labels=labels, showfliers=False)
    plt.title("Numeric Distributions for Common Characteristics")
    plt.xlabel("Characteristic")
    plt.ylabel("Result value (native units; compare within variables only)")
    plt.xticks(rotation=20, ha="right")
    save_current_figure(FIG_DIR / "common_numeric_distributions.png")


def make_correlation_heatmap() -> None:
    matrix_path = PROCESSED_DIR / "florida_wqp_site_characteristic_matrix.csv"
    if not matrix_path.exists() or matrix_path.stat().st_size == 0:
        return
    try:
        matrix = pd.read_csv(matrix_path)
    except pd.errors.EmptyDataError:
        return
    numeric_cols = [col for col in matrix.columns if col != "MonitoringLocationIdentifier"]
    if len(numeric_cols) < 3:
        return
    corr = matrix[numeric_cols].corr(method="spearman", min_periods=30)
    if corr.dropna(how="all").empty:
        return
    plt.figure(figsize=(16, 13))
    if sns is not None:
        ax = sns.heatmap(corr, cmap="vlag", center=0, vmin=-1, vmax=1, cbar_kws={"label": "Spearman correlation"})
        ax.set_xticklabels([wrap_label(label.get_text(), 16) for label in ax.get_xticklabels()], rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels([wrap_label(label.get_text(), 20) for label in ax.get_yticklabels()], rotation=0, fontsize=7)
    else:
        plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
        plt.yticks(range(len(corr.index)), corr.index)
        plt.colorbar(label="Spearman correlation")
    plt.title("Site-Level Spearman Correlation for Candidate Variables")
    save_current_figure(FIG_DIR / "candidate_variable_correlation_heatmap.png")


def build_technique_matrix(
    profile: dict[str, object],
    characteristic_summary: pd.DataFrame,
    site_summary: pd.DataFrame,
    candidate_vars: pd.DataFrame,
) -> pd.DataFrame:
    candidate_count = len(candidate_vars)
    site_count = len(site_summary)
    topics = [
        (
            "CRISP-DM",
            "Strong fit",
            "The project needs disciplined stages because WQP data have uneven coverage, mixed units, censoring, and observational bias.",
            "Raw results, station metadata, EDA summaries, modeling subsets.",
            "Define objective, clean units/dates/non-detects, build site profiles, evaluate and report.",
            "Structured project workflow and defensible final report.",
            "Yes",
        ),
        (
            "Data Understanding",
            "Strong fit",
            "This is the highest-priority requirement and is essential before modeling.",
            "Full results and station files.",
            "Chunked profiling, schema dictionary, missingness, coverage, units, site/date summaries.",
            "EDA tables, figures, and dataset limitations.",
            "Yes",
        ),
        (
            "Data Preparation",
            "Strong fit",
            "Dates, numeric values, station metadata, units, non-detect flags, and sparse records all need preparation.",
            "Selected columns plus station metadata.",
            "Type parsing, joins, unit review, filtering sparse site-characteristic pairs, standardization.",
            "Analysis-ready long file and site-characteristic matrix.",
            "Yes",
        ),
        (
            "Correlation Methods",
            "Possible fit" if candidate_count >= 3 else "Weak fit / not recommended",
            "Correlations are useful only after pivoting repeated unit-consistent variables to site-level profiles.",
            "Site-characteristic matrix with enough co-occurrence.",
            "Median aggregation by site, missingness filtering, standardization, Spearman correlation.",
            "Correlation heatmap for selected variables.",
            "Include if co-occurrence remains adequate.",
        ),
        (
            "Dimensionality Reduction",
            "Strong fit" if candidate_count >= 4 and site_count >= 100 else "Possible fit",
            "PCA can summarize multivariate site profiles after unit and missingness controls.",
            "Standardized site-characteristic matrix.",
            "Impute/filter missing values, scale variables, remove sparse variables/sites.",
            "PCA scatterplots and loadings.",
            "Yes, as a support technique for clustering.",
        ),
        (
            "Frequent Itemset Mining",
            "Possible fit",
            "Can mine co-occurring high/low bins by site type, season, provider, and characteristic group, but numeric binning can be arbitrary.",
            "Binned transactions by site-date or activity.",
            "Discretize carefully, avoid mixing units, set support thresholds.",
            "Common itemsets such as site type + season + high nutrient flag.",
            "Optional only if time allows.",
        ),
        (
            "Association Rules",
            "Possible fit",
            "Rules may be interpretable for categorical/binned conditions, but require careful discretization and support checks.",
            "Same binned transaction table as itemset mining.",
            "Create high/low flags within characteristic/unit; filter rare items.",
            "Association rules with support, confidence, and lift.",
            "Optional, secondary.",
        ),
        (
            "Clustering",
            "Strong fit" if candidate_count >= 3 and site_count >= 100 else "Possible fit",
            "Monitoring sites can be grouped by standardized profiles if enough repeated measurements exist.",
            "Site-level medians for candidate variables plus site metadata for interpretation.",
            "Unit filtering, aggregation, scaling, missingness filtering/imputation.",
            "Site clusters and cluster profiles.",
            "Yes",
        ),
        (
            "Supervised Machine Learning",
            "Possible fit",
            "Needs a defensible target such as high nutrient/conductance/bacteria condition or site type; not all targets are appropriate.",
            "Clean rows or site profiles with a target label.",
            "Target definition, leakage removal, train/test split, encoding/scaling.",
            "Predictive model with evaluation metrics and feature importance.",
            "Include only after choosing a supported target.",
        ),
        (
            "Decision Trees",
            "Strong fit",
            "Interpretable trees are suitable for explaining drivers of high-condition flags or site type differences.",
            "Clean target table with site/date/season/site type/candidate measurements.",
            "Encode categorical features, remove IDs/leakage, handle missingness.",
            "Decision rules and feature importance.",
            "Yes if a target is selected.",
        ),
        (
            "Model Evaluation",
            "Strong fit",
            "Any supervised or clustering result must be evaluated against baselines and stability.",
            "Model predictions, labels, clusters, train/test partitions.",
            "Cross-validation, metrics, baseline models, sensitivity checks.",
            "Confusion matrix/F1/ROC-AUC or RMSE/MAE/R2; silhouette for clusters.",
            "Yes",
        ),
        (
            "Neural Networks",
            "Weak fit / not recommended",
            "The project benefits more from interpretability and data cleaning than high-complexity models.",
            "Large clean labeled table with strong target.",
            "Substantial preprocessing, tuning, regularization, and explanation work.",
            "Black-box predictions with limited interpretability.",
            "No",
        ),
        (
            "Ensemble Methods",
            "Strong fit",
            "Random forest or gradient boosting can benchmark decision trees for a supervised target and provide feature importance.",
            "Clean supervised modeling table.",
            "Leakage removal, encoding, split strategy, hyperparameter control.",
            "Performance benchmark and feature importance.",
            "Yes if supervised target is used.",
        ),
        (
            "Big Data",
            "Possible fit",
            "The raw CSV is large enough to discuss scalable processing, but not large enough to require distributed systems.",
            "Raw CSV and chunked processing logs.",
            "Chunked pandas or optional Dask/Spark comparison.",
            "Scalability discussion, not core model.",
            "Mention briefly, do not center project.",
        ),
        (
            "MapReduce",
            "Weak fit / not recommended",
            "Chunked pandas aggregations solve the needed summaries without MapReduce complexity.",
            "Distributed key-value aggregation tasks.",
            "Rewrite summaries as mappers/reducers.",
            "Counts by key; not better than current workflow.",
            "No",
        ),
        (
            "Spark",
            "Weak fit / not recommended",
            "Useful for much larger WQP exports, but unnecessary for the current file on a normal machine.",
            "Spark dataframe over raw CSV.",
            "Cluster/local Spark setup, schema definitions.",
            "Same summaries at higher setup cost.",
            "No, unless required by instructor.",
        ),
        (
            "Locality-Sensitive Hashing",
            "Weak fit / not recommended",
            "Approximate similarity search is unnecessary unless building a large site-similarity retrieval system.",
            "High-dimensional site profiles.",
            "Vectorization, scaling, approximate nearest neighbor setup.",
            "Similar-site lookup.",
            "No",
        ),
        (
            "Recommender Systems",
            "Weak fit / not recommended",
            "There is no natural user-item preference structure.",
            "Artificial site-characteristic matrix.",
            "Reframe measurements as ratings, which is not scientifically meaningful.",
            "Recommendations of characteristics/sites with weak interpretation.",
            "No",
        ),
        (
            "PageRank",
            "Weak fit / not recommended",
            "The data are not naturally directed graph data.",
            "Constructed graph of sites/characteristics/counties.",
            "Define edges and weights, then rank nodes.",
            "Node rankings that may be hard to justify.",
            "No",
        ),
        (
            "Community Detection in Graphs",
            "Possible fit",
            "A site-similarity graph could be built from standardized profiles, but clustering is simpler and more direct.",
            "Site similarity graph from candidate variables.",
            "Build edges by similarity threshold, validate graph sensitivity.",
            "Communities of similar monitoring sites.",
            "Optional alternative to clustering.",
        ),
        (
            "Mining Data Streams",
            "Weak fit / not recommended",
            "The data are a historical batch export, not a live stream.",
            "Chronologically ordered monitoring feed.",
            "Stream windows and online statistics.",
            "Streaming anomaly/trend summaries.",
            "No",
        ),
    ]
    return pd.DataFrame(
        topics,
        columns=[
            "course_topic",
            "fit_classification",
            "why_it_fits_or_not",
            "input_data_needed",
            "preprocessing_required",
            "expected_output",
            "include_in_final_project",
        ],
    )


def main() -> None:
    print(f"Profiling Florida WQP results from {RESULT_PATH}")
    profile = build_profile()
    print("Writing EDA tables, processed files, figures, and validation summaries")
    write_outputs(profile)
    candidate_count = len(pd.read_csv(EDA_DIR / "wqp_candidate_modeling_variables.csv"))
    site_matrix_shape = pd.read_csv(PROCESSED_DIR / "florida_wqp_site_characteristic_matrix.csv").shape
    date_min = profile["date_min"].date().isoformat() if profile["date_min"] is not None else "unknown"
    date_max = profile["date_max"].date().isoformat() if profile["date_max"] is not None else "unknown"
    print(f"Raw rows: {profile['total_rows']:,}")
    print(f"Processed rows: {profile['processed_rows']:,}")
    print(f"Activity date range: {date_min} to {date_max}")
    print(f"Candidate variables: {candidate_count:,}")
    print(f"Site matrix shape: {site_matrix_shape[0]:,} rows x {site_matrix_shape[1]:,} columns")
    print(f"Profiled {profile['total_rows']:,} rows and {profile['total_columns']} columns.")
    print(f"Wrote EDA tables to {EDA_DIR}")
    print(f"Wrote figures to {FIG_DIR}")
    print(f"Wrote processed files to {PROCESSED_DIR}")


if __name__ == "__main__":
    main()

