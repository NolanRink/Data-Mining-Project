import numpy as np
import pandas as pd
import os

# Step 2: Working columns to retain
WORKING_COLUMNS = [
    'MonitoringLocationIdentifier',
    'MonitoringLocationName',
    'ActivityLocation/LatitudeMeasure',
    'ActivityLocation/LongitudeMeasure',
    'ActivityStartDate',
    'ActivityStartTime/Time',
    'ActivityStartTime/TimeZoneCode',
    'CharacteristicName',
    'ResultMeasureValue',
    'ResultMeasure/MeasureUnitCode',
    'ResultSampleFractionText',
    'ResultDetectionConditionText',
    'MeasureQualifierCode',
    'ResultStatusIdentifier',
    'DetectionQuantitationLimitTypeName',
    'DetectionQuantitationLimitMeasure/MeasureValue',
    'ActivityIdentifier',
    'ResultIdentifier',
    'ProjectIdentifier',
    'ProviderName',
]

# Step 5: Target indicators
TARGET_INDICATORS = [
    'Dissolved oxygen (DO)',
    'pH',
    'Phosphorus',
    'Specific conductance',
    'Dissolved oxygen saturation',
]

# Step 3: Season mapping
def get_season(month):
    if month in (12, 1, 2):
        return 'Winter'
    elif month in (3, 4, 5):
        return 'Spring'
    elif month in (6, 7, 8):
        return 'Summer'
    else:
        return 'Fall'


def load_data(file_path):
    df = pd.read_csv(file_path, low_memory=False)
    return df


def preProcess(df):
    # Step 2: Select working columns (keep only those that exist in df)
    cols = [c for c in WORKING_COLUMNS if c in df.columns]
    df = df[cols].copy()

    # Step 3: Parse dates and time variables
    df['activity_date'] = pd.to_datetime(df['ActivityStartDate'], errors='coerce')
    df['year'] = df['activity_date'].dt.year
    df['month'] = df['activity_date'].dt.month
    df['season'] = df['month'].map(get_season)

    # Step 4: Parse numeric result values; keep raw text for audit
    df['result_value_num'] = pd.to_numeric(df['ResultMeasureValue'], errors='coerce')

    # Step 5: Filter to target indicators only
    df = df[df['CharacteristicName'].isin(TARGET_INDICATORS)].copy()

    # Step 6: Reconcile units
    # Treat umho/cm and uS/cm as equivalent for specific conductance; no value conversion needed
    # pH is unitless; dissolved oxygen stays in mg/L; phosphorus stays in mg/L; DO saturation in %
    # Create phosphorus fraction features
    df['Phosphorus_Total'] = np.where(
        (df['CharacteristicName'] == 'Phosphorus') &
        (df['ResultSampleFractionText'].str.strip().str.lower() == 'total'),
        df['result_value_num'],
        np.nan,
    )
    df['Phosphorus_Dissolved'] = np.where(
        (df['CharacteristicName'] == 'Phosphorus') &
        (df['ResultSampleFractionText'].str.strip().str.lower() == 'dissolved'),
        df['result_value_num'],
        np.nan,
    )

    # Step 7: Duplicate check — flag rows with duplicate ResultIdentifier
    df['duplicate_result_id_flag'] = df.duplicated(subset=['ResultIdentifier'], keep=False)

    # Step 9: Outlier/data-quality flags using IQR per indicator
    df['outlier_flag'] = False
    for indicator in TARGET_INDICATORS:
        mask = df['CharacteristicName'] == indicator
        values = df.loc[mask, 'result_value_num'].dropna()
        if len(values) < 4:
            continue
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_mask = mask & (
            (df['result_value_num'] < lower) | (df['result_value_num'] > upper)
        )
        df.loc[outlier_mask, 'outlier_flag'] = True

    return df


def main():

    base = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base, 'resultphyschem', 'resultphyschem.csv')
    df = load_data(file_path)

    processed_df = preProcess(df)


    os.makedirs(os.path.join(base, 'outputs', 'tables'), exist_ok=True)
    out_path = os.path.join(base, 'outputs', 'tables', 'cleaned_primary_indicators.csv')
    processed_df.to_csv(out_path, index=False)
    print(f"Saved cleaned table: {out_path}  ({len(processed_df)} rows)")


if __name__ == "__main__":
    main()
