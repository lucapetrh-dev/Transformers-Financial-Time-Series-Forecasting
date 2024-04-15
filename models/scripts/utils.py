import os
import warnings

import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import OneHotEncoder

def create_sequences(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps - 1):
        end = i + n_steps
        seq_x, seq_y = data.iloc[i:end, :].values, data.iloc[end]["close"]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def load_csv(file_path):

    df = pd.read_csv(file_path)
    df['time'] = pd.to_datetime(df['time'])

    return df

def split_date(df, date):
    """
    Splits a datetime column into separate day, month, and year columns in the DataFrame.

    Parameters:
    - df: pandas DataFrame containing the datetime column to split.
    - date: str, the name of the datetime column to split. Default is 'time'.

    Returns:
    - A DataFrame with the original datetime column split into separate day, month, and year columns.
    """

    df = df.sort_values(by='time')

    # Extract and create separate day, month, and year columns
    df['day'] = df[date].dt.day
    df['month'] = df[date].dt.month
    df['year'] = df[date].dt.year

    return df

def trigonometric_date_encoding(df: pd.DataFrame, column: str = "time") -> pd.DataFrame:
    """Encode date as sin and cos of the day of the week from a date object.

    Args:
        df (pd.DataFrame): The dataframe.
        column (str, optional): The column name with the date to encode. Defaults to "f_1".

    Returns:
        pd.DataFrame: The dataframe with the encoded date.
            The new columns are called sin_date and cos_date.
            The original column is not dropped.
    """
    # Convert the column to datetime
    df[column] = pd.to_datetime(df[column], format="%d-%m-%Y")

    # Extract the day of the week (0 = Monday, 6 = Sunday)
    day_of_week = df[column].dt.dayofweek

    # Calculate sin and cos
    date_sin = np.sin(day_of_week * (2.0 * np.pi / 7.0))
    date_cos = np.cos(day_of_week * (2.0 * np.pi / 7.0))

    # Create a DataFrame with the new columns
    encoded_dates = pd.DataFrame({"sin_date": date_sin, "cos_date": date_cos})

    # Concatenate the new columns with the original dataframe
    result_df = pd.concat([df, encoded_dates], axis=1)

    return result_df


def drop_columns(df, threshold=50.0):
    """
    Drops columns from the DataFrame if they have more than a specified percentage of missing values
    or if all values in the column are the same.

    Parameters:
    - df: pandas DataFrame from which to remove columns.
    - threshold: float, the percentage threshold of missing values above which columns will be dropped. Default is 50.0.

    Returns:
    - df_cleaned: pandas DataFrame after columns with excessive missing values or identical values have been removed.
    """
    # Calculate the percentage of missing values for each column
    percentage_missing = df.isnull().sum() * 100 / len(df)

    # Identify columns that exceed the threshold for missing values
    columns_to_drop_missing = percentage_missing[percentage_missing > threshold].index

    # Identify columns where all values are the same
    columns_to_drop_unique = [col for col in df.columns if df[col].nunique() <= 1]

    # Combine the two lists of columns to drop, ensuring uniqueness
    columns_to_drop = set(columns_to_drop_missing).union(set(columns_to_drop_unique))

    # Drop these columns
    df_cleaned = df.drop(columns=columns_to_drop)

    return df_cleaned

def display_missing_values(df):
    """
    Calculates and displays the number and percentage of missing values for each column in a DataFrame.

    Parameters:
    - df: pandas DataFrame to analyze for missing values.

    Returns:
    - A DataFrame displaying columns with missing values, the number of missing values, and the percentage of total missing.
    """
    # Calculating the number and percentage of missing values for each column
    missing_values = df.isnull().sum()
    percentage_missing = (missing_values / len(df)) * 100

    # Creating a DataFrame to display the number and percentage of missing values
    missing_values_df = pd.DataFrame({'Number of Missing Values': missing_values, 'Percentage': percentage_missing})

    # Filtering out columns that have no missing values to focus on those that do
    missing_values_df = missing_values_df[missing_values_df['Percentage'] > 0].sort_values(by='Percentage', ascending=False)

    return missing_values_df
def impute_rolling_median(df, window_size=5):
    """
    Imputes missing values using a rolling median calculated over a specified window size.
    For initial NaN values where the rolling median cannot be computed, it uses the first available value of the column.
    This method is time-sensitive and uses surrounding data points to calculate the median.

    Parameters:
    - df: pandas DataFrame with missing values to impute.
    - window_size: int, the size of the rolling window to use for computing the median.

    Returns:
    - A DataFrame with missing values imputed using the rolling median and the first available value where necessary.
    """
    # Ensure the DataFrame is sorted by time if it has a time column
    if 'time' in df.columns:
        df = df.sort_values(by='time')

    numeric_columns = df.select_dtypes(include=['number']).columns
    for column in numeric_columns:
        # Compute rolling median with a specified window, centering the window and minimizing edge effects
        rolling_median = df[column].rolling(window=window_size, center=True, min_periods=1).median()

        # Use the first available non-NaN value for initial NaN values
        first_non_nan = df[column].first_valid_index()
        if first_non_nan is not None:
            df[column].fillna(method='bfill', inplace=True)  # Backfill to address initial NaNs
            df[column].fillna(rolling_median, inplace=True)  # Then fill remaining NaNs with rolling median
        else:
            # If the column is entirely NaN, no action is taken
            continue

    return df

def save_to_csv(df, file_path):
    """
    Saves the given DataFrame as a CSV file at the specified path.

    """
    df.to_csv(file_path, index=False)

def create_lags(df, n_lags):
    def fill_with_first_close(lag_df, n_lags):
        for lag in range(1, n_lags + 1):
            first_valid_index = lag_df["close"].first_valid_index()
            first_valid_value = (
                lag_df.loc[first_valid_index, "close"]
                if first_valid_index is not None
                else 0
            )
            lag_df[f"lag_{lag}"] = lag_df[f"lag_{lag}"].fillna(first_valid_value)
        return lag_df

    lag_df = df.copy()

    for lag in range(1, n_lags + 1):
        lag_df[f"lag_{lag}"] = lag_df["close"].shift(lag)

    return fill_with_first_close(lag_df, n_lags)

def add_seasonality(df):
    def categorize_month(month):
        if month in [11, 12, 1, 2, 4, 5, 8]:
            return "Bullish"
        elif month in [3, 6, 7, 10]:
            return "Bearish"
        else:
            return "Normal"

    df["Month_Category"] = df["month"].apply(categorize_month)

    encoder = OneHotEncoder()
    encoded_data = encoder.fit_transform(df[["Month_Category"]]).toarray()

    encoded_df = pd.DataFrame(
        encoded_data, columns=encoder.get_feature_names_out(["Month_Category"])
    )

    df_final = pd.concat([df, encoded_df], axis=1)
    df_final.drop(["Month_Category"], axis=1, inplace=True)
    return df_final

def frac_diff_stationarity(train, test):
    # Make a copy of the train data inside the function to avoid modifying the original dataframe
    train_internal_copy = train.copy()

    fd = FracdiffStat()
    fd.fit(train_internal_copy[["close"]].values)

    # Replace the 'Close' column with the transformed data in the copy
    train_internal_copy["close"] = fd.transform(train_internal_copy[["close"]].values)
    test["close"] = fd.transform(test[["close"]].values)

    # Return the modified copy and test
    return train_internal_copy, test


def split_data_frame(df, train_frac=0.7, val_frac=0.2):
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    train_df = df[:train_end]
    val_df = df[train_end:val_end]
    test_df = df[val_end:]
    return train_df, val_df, test_df


def apply_moving_average_for_roc(dataframe, ma_type="ema", ma_window=50, roc_window=20):
    def bin_roc_adjusted(roc_value):
        if pd.isna(roc_value):
            return "Unknown"  # Handling NaN values separately
        elif roc_value > 10:
            return "Very High Positive"
        elif roc_value > 5:
            return "High Positive"
        elif roc_value > 1:
            return "Low Positive"
        elif roc_value > -1:
            return "Neutral"
        elif roc_value > -5:
            return "Low Negative"
        else:
            return "High Negative"

    df = dataframe.copy()

    if ma_type == "ma":
        # Calculate the 50-Day Moving Average
        df["50-Day MA"] = df["close"].rolling(window=ma_window).mean()
        # Calculate the Rate of Change for the 50-Day Moving Average
        df["Rate of Change"] = df["50-Day MA"].pct_change(periods=roc_window) * 100
    elif ma_type == "ema":
        # Calculate the 50-Day Exponential Moving Average
        df["50-Day MA"] = df["close"].ewm(span=ma_window, adjust=False).mean()
        # Calculate the Rate of Change for the 50-Day Exponential Moving Average
        df["Rate of Change"] = df["50-Day MA"].pct_change(periods=roc_window) * 100
    else:
        raise ValueError(
            "Invalid ma_type. Choose 'ma' for Moving Average or 'ema' for Exponential Moving Average"
        )

    # Bin the Rate of Change
    df["ROC"] = df["Rate of Change"].apply(bin_roc_adjusted)

    # Use BinaryEncoder to encode the 'ROC' column
    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=FutureWarning)
        from category_encoders import BinaryEncoder

        encoder = BinaryEncoder(cols=["ROC"], drop_invariant=True)
        df_encoded = encoder.fit_transform(df[["ROC"]])

    # Concatenate the encoded 'ROC' column with the original dataframe
    df = pd.concat([df, df_encoded], axis=1)
    df.drop(columns=["ROC", "Rate of Change", "50-Day MA"], inplace=True)

    # Return the dataframe with the encoded 'ROC' column added

    return df

def simplify_df(df):
    """
    Keeps only specified columns in the DataFrame.

    Parameters:
    - df: pandas DataFrame from which to select columns.

    Returns:
    - df_simplified: pandas DataFrame with only specified columns retained.
    """
    # List of columns to keep
    columns_to_keep = ['timestamp', 'time', 'open', 'close', 'high', 'low', 'volume_24h', 'circulating_supply', 'day', 'month', 'year']

    # Select only the specified columns
    df_simplified = df[columns_to_keep].copy()

    return df_simplified
