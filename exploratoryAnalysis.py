import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = os.path.dirname(os.path.abspath(__file__))
CLEANED_CSV = os.path.join(BASE, 'outputs', 'tables', 'cleaned_primary_indicators.csv')
FIGURES_DIR = os.path.join(BASE, 'outputs', 'figures')
TABLES_DIR  = os.path.join(BASE, 'outputs', 'tables')

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(TABLES_DIR,  exist_ok=True)

TARGET_INDICATORS = [
    'Dissolved oxygen (DO)',
    'pH',
    'Phosphorus',
    'Specific conductance',
    'Dissolved oxygen saturation',
]

SEASON_ORDER = ['Winter', 'Spring', 'Summer', 'Fall']

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_cleaned(path=CLEANED_CSV):
    df = pd.read_csv(path, low_memory=False)
    df['activity_date'] = pd.to_datetime(df['activity_date'], errors='coerce')
    return df

# ---------------------------------------------------------------------------
# Section D – Overall summaries
# ---------------------------------------------------------------------------

def overall_summaries(df):
    print("=" * 60)
    print("OVERALL SUMMARIES")
    print("=" * 60)
    print(f"Total rows          : {len(df):,}")
    print(f"Unique sites        : {df['MonitoringLocationIdentifier'].nunique():,}")
    print(f"Years covered       : {sorted(df['year'].dropna().astype(int).unique().tolist())}")
    print(f"Unique characteristics in cleaned data: {df['CharacteristicName'].nunique()}")
    print(f"Numeric result rows : {df['result_value_num'].notna().sum():,}")
    print(f"Missing result rows : {df['result_value_num'].isna().sum():,}")
    print()

    # Rows by year
    rows_by_year = df.groupby('year').size().reset_index(name='row_count')
    print("Rows by year (NOTE: 2023-2025 coverage is sparse):")
    print(rows_by_year.to_string(index=False))
    print()

    # Rows by indicator
    rows_by_indicator = df.groupby('CharacteristicName').agg(
        row_count=('result_value_num', 'size'),
        numeric_count=('result_value_num', 'count'),
        site_count=('MonitoringLocationIdentifier', 'nunique'),
    ).reset_index()
    print("Rows by indicator:")
    print(rows_by_indicator.to_string(index=False))
    print()

    # Column population summary
    total = len(df)
    pop = (df.notna().sum() / total * 100).round(1).sort_values(ascending=False)
    print("Column population % (non-null):")
    print(pop.to_string())
    print()

    # Quality/status field summaries
    for col in ['ResultStatusIdentifier', 'ResultDetectionConditionText', 'MeasureQualifierCode']:
        if col in df.columns:
            vc = df[col].value_counts(dropna=False).head(10)
            print(f"  {col}:\n{vc.to_string()}\n")

    rows_by_year.to_csv(os.path.join(TABLES_DIR, 'eda_rows_by_year.csv'), index=False)
    rows_by_indicator.to_csv(os.path.join(TABLES_DIR, 'eda_rows_by_indicator.csv'), index=False)

    # Rows-by-year bar chart
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(rows_by_year['year'].astype(int), rows_by_year['row_count'], color='steelblue')
    ax.set_xlabel('Year')
    ax.set_ylabel('Row count')
    ax.set_title('Rows by Year\n(NOTE: 2023–2025 coverage is much sparser)')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'rows_by_year.png'), dpi=150)
    plt.close(fig)
    print("Saved: rows_by_year.png")

# ---------------------------------------------------------------------------
# Section D – Missingness and quality
# ---------------------------------------------------------------------------

def missingness_quality(df):
    print("=" * 60)
    print("MISSINGNESS AND QUALITY")
    print("=" * 60)

    # Missing values bar chart
    key_cols = [
        'MonitoringLocationIdentifier',
        'ActivityLocation/LatitudeMeasure',
        'ActivityLocation/LongitudeMeasure',
        'activity_date',
        'CharacteristicName',
        'ResultMeasureValue',
        'result_value_num',
        'ResultMeasure/MeasureUnitCode',
        'ResultSampleFractionText',
        'ResultStatusIdentifier',
        'ResultDetectionConditionText',
        'MeasureQualifierCode',
        'ResultIdentifier',
        'duplicate_result_id_flag',
        'outlier_flag',
    ]
    key_cols = [c for c in key_cols if c in df.columns]
    missing_pct = (df[key_cols].isna().sum() / len(df) * 100).round(1).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    missing_pct.plot.barh(ax=ax, color='salmon')
    ax.set_xlabel('% missing')
    ax.set_title('Missing values by key column')
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'missingness_key_columns.png'), dpi=150)
    plt.close(fig)
    print("Saved: missingness_key_columns.png")

    # Non-numeric / detection condition counts by characteristic
    non_numeric = df[df['result_value_num'].isna()].groupby('CharacteristicName').agg(
        non_numeric_rows=('ResultMeasureValue', 'size'),
        detection_condition_non_null=('ResultDetectionConditionText', lambda x: x.notna().sum()),
    ).reset_index()
    print("\nNon-numeric result rows by characteristic:")
    print(non_numeric.to_string(index=False))
    non_numeric.to_csv(os.path.join(TABLES_DIR, 'eda_non_numeric_by_characteristic.csv'), index=False)

    # Duplicate key summary
    dup_count = df['duplicate_result_id_flag'].sum() if 'duplicate_result_id_flag' in df.columns else 'N/A'
    print(f"\nRows flagged as duplicate ResultIdentifier: {dup_count}")

# ---------------------------------------------------------------------------
# Section D – Indicator distributions
# ---------------------------------------------------------------------------

def indicator_distributions(df):
    print("=" * 60)
    print("INDICATOR DISTRIBUTIONS")
    print("=" * 60)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for i, indicator in enumerate(TARGET_INDICATORS):
        sub = df[(df['CharacteristicName'] == indicator) & df['result_value_num'].notna()]
        ax = axes[i]
        ax.hist(sub['result_value_num'], bins=50, color='steelblue', edgecolor='none')
        ax.set_title(indicator, fontsize=9)
        ax.set_xlabel('Value')
        ax.set_ylabel('Count')
        n = len(sub)
        med = sub['result_value_num'].median()
        ax.axvline(med, color='red', linestyle='--', linewidth=1, label=f'median={med:.2f}')
        ax.legend(fontsize=7)
        print(f"  {indicator}: n={n:,}, median={med:.3f}, "
              f"min={sub['result_value_num'].min():.3f}, max={sub['result_value_num'].max():.3f}")

    # Hide unused subplot
    for j in range(len(TARGET_INDICATORS), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('Indicator Distributions (all records including flagged outliers)', fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'indicator_distributions.png'), dpi=150)
    plt.close(fig)
    print("Saved: indicator_distributions.png\n")

    # Boxplots for each indicator (outliers shown)
    fig, axes = plt.subplots(1, len(TARGET_INDICATORS), figsize=(16, 5))
    for i, indicator in enumerate(TARGET_INDICATORS):
        sub = df[(df['CharacteristicName'] == indicator) & df['result_value_num'].notna()]
        axes[i].boxplot(sub['result_value_num'], vert=True, patch_artist=True,
                        boxprops=dict(facecolor='lightblue'),
                        flierprops=dict(marker='.', markersize=2, alpha=0.4))
        axes[i].set_title(indicator, fontsize=8)
        axes[i].set_xticklabels([])
    fig.suptitle('Indicator Boxplots (outliers shown)', fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'indicator_boxplots.png'), dpi=150)
    plt.close(fig)
    print("Saved: indicator_boxplots.png\n")

    # Phosphorus by fraction
    phos = df[(df['CharacteristicName'] == 'Phosphorus') & df['result_value_num'].notna()].copy()
    if not phos.empty and 'ResultSampleFractionText' in phos.columns:
        fractions = phos['ResultSampleFractionText'].fillna('Unknown').str.strip()
        fig, ax = plt.subplots(figsize=(7, 4))
        groups = {f: phos.loc[fractions == f, 'result_value_num'] for f in fractions.unique()}
        ax.boxplot(groups.values(), labels=groups.keys(), patch_artist=True,
                   flierprops=dict(marker='.', markersize=2, alpha=0.4))
        ax.set_title('Phosphorus by Sample Fraction')
        ax.set_ylabel('mg/L')
        fig.tight_layout()
        fig.savefig(os.path.join(FIGURES_DIR, 'phosphorus_by_fraction.png'), dpi=150)
        plt.close(fig)
        print("Saved: phosphorus_by_fraction.png\n")

# ---------------------------------------------------------------------------
# Section D – Seasonal patterns
# ---------------------------------------------------------------------------

def seasonal_patterns(df):
    print("=" * 60)
    print("SEASONAL PATTERNS")
    print("=" * 60)

    df_clean = df[df['result_value_num'].notna() & df['season'].notna()].copy()
    df_clean['season'] = pd.Categorical(df_clean['season'], categories=SEASON_ORDER, ordered=True)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    for i, indicator in enumerate(TARGET_INDICATORS):
        sub = df_clean[df_clean['CharacteristicName'] == indicator]
        season_groups = [sub.loc[sub['season'] == s, 'result_value_num'] for s in SEASON_ORDER]
        ax = axes[i]
        bp = ax.boxplot(season_groups, labels=SEASON_ORDER, patch_artist=True,
                        flierprops=dict(marker='.', markersize=2, alpha=0.4))
        colors = ['#aec6cf', '#77dd77', '#fdfd96', '#ffb347']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        ax.set_title(indicator, fontsize=9)
        ax.set_ylabel('Value')
        counts = [len(g) for g in season_groups]
        for j, (s, c) in enumerate(zip(SEASON_ORDER, counts)):
            ax.text(j + 1, ax.get_ylim()[0], f'n={c}', ha='center', va='bottom', fontsize=6)

    for j in range(len(TARGET_INDICATORS), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('Seasonal Boxplots by Indicator\n'
                 '(NOTE: 2023–2025 rows much sparser; treat as coverage-aware context only)',
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'seasonal_boxplots.png'), dpi=150)
    plt.close(fig)
    print("Saved: seasonal_boxplots.png\n")

    # Median seasonal summary table
    seasonal_summary = df_clean.groupby(['CharacteristicName', 'season'])['result_value_num'].agg(
        median='median', count='count'
    ).reset_index()
    seasonal_summary.to_csv(os.path.join(TABLES_DIR, 'eda_seasonal_summary.csv'), index=False)
    print("Seasonal median summary:")
    print(seasonal_summary.to_string(index=False))
    print()

    # Year-context medians with row-count caveat
    year_summary = df_clean.groupby(['CharacteristicName', 'year'])['result_value_num'].agg(
        median='median', count='count'
    ).reset_index()
    year_summary.to_csv(os.path.join(TABLES_DIR, 'eda_year_context_summary.csv'), index=False)
    print("Year-context medians (use with caution for 2023-2025 — sparse coverage):")
    print(year_summary.to_string(index=False))
    print()

# ---------------------------------------------------------------------------
# Section D – Site-level comparisons
# ---------------------------------------------------------------------------

def site_level_comparisons(df, min_obs=5):
    print("=" * 60)
    print(f"SITE-LEVEL COMPARISONS (min_obs={min_obs})")
    print("=" * 60)

    df_clean = df[df['result_value_num'].notna()].copy()
    site_summary = df_clean.groupby(['CharacteristicName', 'MonitoringLocationIdentifier']).agg(
        median_value=('result_value_num', 'median'),
        obs_count=('result_value_num', 'count'),
        lat=('ActivityLocation/LatitudeMeasure', 'first'),
        lon=('ActivityLocation/LongitudeMeasure', 'first'),
    ).reset_index()

    # Filter to sites with sufficient observations
    site_filtered = site_summary[site_summary['obs_count'] >= min_obs].copy()

    site_filtered.to_csv(os.path.join(TABLES_DIR, 'eda_site_summary.csv'), index=False)

    for indicator in TARGET_INDICATORS:
        sub = site_filtered[site_filtered['CharacteristicName'] == indicator].sort_values(
            'median_value', ascending=False
        )
        if sub.empty:
            continue
        print(f"\n  {indicator} — top 5 sites by median value:")
        print(sub[['MonitoringLocationIdentifier', 'median_value', 'obs_count']].head(5).to_string(index=False))
        print(f"  {indicator} — bottom 5 sites by median value:")
        print(sub[['MonitoringLocationIdentifier', 'median_value', 'obs_count']].tail(5).to_string(index=False))

    print()

    return site_filtered

# ---------------------------------------------------------------------------
# Section D – Spatial plots
# ---------------------------------------------------------------------------

def spatial_plots(site_summary):
    print("=" * 60)
    print("SPATIAL PLOTS")
    print("=" * 60)

    lat_col = 'ActivityLocation/LatitudeMeasure'
    lon_col = 'ActivityLocation/LongitudeMeasure'

    # Check coordinate columns exist (renamed by preProcess or still original)
    if lat_col not in site_summary.columns:
        lat_col = 'lat'
    if lon_col not in site_summary.columns:
        lon_col = 'lon'

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    for i, indicator in enumerate(TARGET_INDICATORS):
        sub = site_summary[
            (site_summary['CharacteristicName'] == indicator) &
            site_summary[lat_col].notna() &
            site_summary[lon_col].notna()
        ]
        ax = axes[i]
        if sub.empty:
            ax.set_title(f'{indicator}\n(no coordinate data)')
            continue
        sc = ax.scatter(
            sub[lon_col], sub[lat_col],
            c=sub['median_value'], cmap='RdYlGn', s=15, alpha=0.7,
        )
        plt.colorbar(sc, ax=ax, shrink=0.7)
        ax.set_title(indicator, fontsize=8)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')

    for j in range(len(TARGET_INDICATORS), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('Site-level Median Indicator Values by Coordinate\n'
                 '(coordinate scatterplots — not GIS boundary maps)', fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'spatial_indicator_maps.png'), dpi=150)
    plt.close(fig)
    print("Saved: spatial_indicator_maps.png\n")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Loading cleaned data from: {CLEANED_CSV}\n")
    df = load_cleaned()

    overall_summaries(df)
    missingness_quality(df)
    indicator_distributions(df)
    seasonal_patterns(df)
    site_summary = site_level_comparisons(df)
    spatial_plots(site_summary)

    print("EDA complete. Figures saved to:", FIGURES_DIR)
    print("Tables saved to:", TABLES_DIR)


if __name__ == '__main__':
    main()
