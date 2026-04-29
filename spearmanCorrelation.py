import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = os.path.dirname(os.path.abspath(__file__))
CLEANED_CSV  = os.path.join(BASE, 'outputs', 'tables', 'cleaned_primary_indicators.csv')
FIGURES_DIR  = os.path.join(BASE, 'outputs', 'figures')
TABLES_DIR   = os.path.join(BASE, 'outputs', 'tables')

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(TABLES_DIR,  exist_ok=True)

# Minimum observations a site must have for a feature to count
MIN_OBS_PER_FEATURE = 3

# Features used in the wide site profile (phosphorus fractions kept separate)
WIDE_FEATURES = [
    'Dissolved oxygen (DO)',
    'pH',
    'Specific conductance',
    'Dissolved oxygen saturation',
    'Phosphorus_Total',
    'Phosphorus_Dissolved',
]

# ---------------------------------------------------------------------------
# Build wide site-level profile
# ---------------------------------------------------------------------------

def build_site_profile(df):
    """
    Pivot the cleaned row-level table into a wide site profile where each
    column is the site-level median for one indicator feature.
    Phosphorus fractions come from the pre-computed Phosphorus_Total /
    Phosphorus_Dissolved columns; all other indicators use CharacteristicName.
    Sites must have at least MIN_OBS_PER_FEATURE observations for a feature
    to receive a non-NaN entry.
    """
    df = df[df['result_value_num'].notna()].copy()
    site_col = 'MonitoringLocationIdentifier'

    frames = []

    # Standard indicators (not phosphorus fractions)
    standard = [f for f in WIDE_FEATURES if f not in ('Phosphorus_Total', 'Phosphorus_Dissolved')]
    for indicator in standard:
        sub = df[df['CharacteristicName'] == indicator]
        agg = sub.groupby(site_col)['result_value_num'].agg(
            median='median', count='count'
        ).reset_index()
        agg.loc[agg['count'] < MIN_OBS_PER_FEATURE, 'median'] = np.nan
        agg = agg.rename(columns={'median': indicator}).drop(columns='count')
        frames.append(agg.set_index(site_col))

    # Phosphorus fractions
    for frac_col in ('Phosphorus_Total', 'Phosphorus_Dissolved'):
        if frac_col not in df.columns:
            continue
        sub = df[df[frac_col].notna()]
        agg = sub.groupby(site_col)[frac_col].agg(
            median='median', count='count'
        ).reset_index()
        agg.loc[agg['count'] < MIN_OBS_PER_FEATURE, 'median'] = np.nan
        agg = agg.rename(columns={'median': frac_col}).drop(columns='count')
        frames.append(agg.set_index(site_col))

    if not frames:
        raise ValueError("No feature frames built — check that CLEANED_CSV exists and has data.")

    wide = pd.concat(frames, axis=1).reset_index()

    # Keep only features that are actually present as columns
    present = [f for f in WIDE_FEATURES if f in wide.columns]
    wide = wide[[site_col] + present]

    print(f"Wide site profile: {len(wide)} sites, {len(present)} features")
    print(f"Complete-case sites (all features non-NaN): "
          f"{wide[present].dropna().shape[0]}")
    return wide, present


# ---------------------------------------------------------------------------
# Spearman correlation
# ---------------------------------------------------------------------------

def spearman_correlation(wide, features):
    """
    Compute the pairwise Spearman correlation matrix over complete cases
    (sites with non-NaN values for all features).  Returns the correlation
    DataFrame and the p-value DataFrame.
    """
    data = wide[features].dropna()
    n_sites = len(data)
    print(f"\nSpearman correlation computed on {n_sites} complete-case sites.\n")
    if n_sites < 5:
        print("WARNING: fewer than 5 complete-case sites — correlation results "
              "unreliable.  Consider reducing MIN_OBS_PER_FEATURE or the "
              "number of required features.")

    n = len(features)
    corr_matrix = np.ones((n, n))
    pval_matrix = np.ones((n, n))

    for i, f1 in enumerate(features):
        for j, f2 in enumerate(features):
            if i == j:
                continue
            rho, pval = stats.spearmanr(data[f1], data[f2])
            corr_matrix[i, j] = rho
            pval_matrix[i, j] = pval

    corr_df = pd.DataFrame(corr_matrix, index=features, columns=features)
    pval_df = pd.DataFrame(pval_matrix, index=features, columns=features)
    return corr_df, pval_df, n_sites


# ---------------------------------------------------------------------------
# Heatmap
# ---------------------------------------------------------------------------

def plot_heatmap(corr_df, pval_df, n_sites):
    """Save a Spearman correlation heatmap with significance markers."""
    fig, ax = plt.subplots(figsize=(9, 7))
    n = len(corr_df)

    im = ax.imshow(corr_df.values, vmin=-1, vmax=1, cmap='RdBu_r', aspect='auto')
    plt.colorbar(im, ax=ax, label='Spearman ρ')

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(corr_df.columns, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(corr_df.index, fontsize=8)

    # Annotate each cell with ρ value; mark p < 0.05 with *
    for i in range(n):
        for j in range(n):
            rho  = corr_df.values[i, j]
            pval = pval_df.values[i, j]
            sig  = '*' if (i != j and pval < 0.05) else ''
            text_color = 'white' if abs(rho) > 0.6 else 'black'
            ax.text(j, i, f'{rho:.2f}{sig}', ha='center', va='center',
                    fontsize=7, color=text_color)

    ax.set_title(
        f'Spearman Correlation — Site-level Median Indicator Profiles\n'
        f'(n={n_sites} complete-case sites; * p < 0.05)',
        fontsize=10,
    )
    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, 'spearman_correlation_heatmap.png')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Loading cleaned data from: {CLEANED_CSV}\n")
    df = pd.read_csv(CLEANED_CSV, low_memory=False)

    wide, features = build_site_profile(df)

    # Save wide profile
    wide_path = os.path.join(TABLES_DIR, 'site_wide_profile.csv')
    wide.to_csv(wide_path, index=False)
    print(f"Saved wide site profile: {wide_path}\n")

    corr_df, pval_df, n_sites = spearman_correlation(wide, features)

    # Save correlation and p-value tables
    corr_path = os.path.join(TABLES_DIR, 'spearman_correlation_matrix.csv')
    pval_path = os.path.join(TABLES_DIR, 'spearman_pvalue_matrix.csv')
    corr_df.to_csv(corr_path)
    pval_df.to_csv(pval_path)
    print(f"Saved correlation matrix: {corr_path}")
    print(f"Saved p-value matrix    : {pval_path}\n")

    print("Spearman Correlation Matrix:")
    print(corr_df.round(3).to_string())
    print()

    plot_heatmap(corr_df, pval_df, n_sites)
    print("\nDone.")


if __name__ == '__main__':
    main()
