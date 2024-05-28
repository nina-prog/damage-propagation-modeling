""" This module contains functions for data cleaning. """
import pandas as pd
from typing import Union, List

from src.logger import setup_logger
from src.outlier_detection import remove_outliers
from src.rolling_window_creator import calculate_RUL

logger = setup_logger(__name__, level='DEBUG')  # Change the level to 'DEBUG' to see more information


def identify_missing_values(df: pd.DataFrame, threshold: float = 0.1, drop: bool = False) -> Union[pd.DataFrame, List[str]]:
    """
    Identify features with missing values above a certain threshold.

    :param df: DataFrame with features to investigate.
    :param threshold: Threshold for the relative amount of missing values.
    :param drop: Boolean to indicate whether to drop the features from the DataFrame.
    :return: DataFrame or list
    """
    missing_values = df.isnull().sum()
    total_rows = len(df)
    relative_missing_values = missing_values / total_rows
    features_above_threshold = relative_missing_values[relative_missing_values > threshold].index.tolist()

    logger.debug(
        f"Found {len(features_above_threshold)} features with missing values above the threshold of {threshold}.")

    if drop:
        df.drop(columns=features_above_threshold, inplace=True)
        return df
    else:
        return features_above_threshold


def identify_single_unique_features(df: pd.DataFrame, drop: bool = False) -> Union[pd.DataFrame, List[str]]:
    """
    Identify features with only a single unique value.

    :param df: DataFrame with features to investigate.
    :param drop: Boolean to indicate whether to drop the features from the DataFrame.
    :return: DataFrame or list
    """
    single_unique_features = [col for col in df.columns if df[col].nunique(dropna=True) == 1]

    logger.debug(f"Found {len(single_unique_features)} features with only a single unique value: {single_unique_features}")

    if drop:
        df.drop(columns=single_unique_features, inplace=True)
        return df
    else:
        return single_unique_features


def format_dtype(df: pd.DataFrame) -> pd.DataFrame:
    """
    Formats the data types of columns in a pandas DataFrame.

    :param df: The input dataframe to be formatted.
    :type df: pandas.DataFrame

    :return: The formatted dataframe.
    :rtype: pandas.DataFrame
    """
    # categorical values
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    df[cat_cols] = df[cat_cols].astype('category')

    logger.debug(f"Found {len(cat_cols)} categorical columns: {cat_cols}")

    return df


def get_uncorrelated_features(df: pd.DataFrame, threshold: float = 0.2, target: str = None) -> List[str]:
    """
    Get features from the input DataFrame that are not highly correlated with the target column.

    :param df: The input DataFrame.
    :type df: pd.DataFrame
    :param threshold: The correlation threshold to use. Default is 0.9.
                      Features with an absolute correlation coefficient less than this value with the target are considered uncorrelated.
    :type threshold: float
    :param target: The target column name. Default is None.
    :type target: str

    :return: The list of uncorrelated feature names.
    :rtype: List[str]
    """

    if target and target not in df.columns:
        raise ValueError(f"Target column '{target}' is not in the DataFrame.")

    corr_matrix = df.corr().abs()

    if target:
        target_corr = corr_matrix[target].drop(target)
        # take the absolute value of the correlation with the target
        target_corr = target_corr.abs()
        uncorrelated_features = target_corr[target_corr < threshold].index.tolist()
    else:
        uncorrelated_features = df.columns.tolist()

    logger.debug(f"Found {len(uncorrelated_features)} uncorrelated features with a correlation threshold of {threshold}: {uncorrelated_features}")

    return uncorrelated_features


# TODO: Refactor the process of data cleaning in a class
def clean_data(train_df: pd.DataFrame, test_df: pd.DataFrame, method: Union[str, None] = 'winsorize', ignore_columns: List[str] = None, contamination: float = 0.05, threshold_sd: float = 0.8, threshold_missing: float = 0.1, soft_drop: bool = False, threshold_corr: float = 0.9) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clean the input train and test DataFrames. The cleaning steps include handling missing values, duplicates, outliers, single unique values, and uncorrelated features.

    :param train_df: The input train DataFrame to clean.
    :type train_df: pd.DataFrame
    :param test_df: The input test DataFrame to clean.
    :type test_df: pd.DataFrame
    :param method: The outlier detection method to use. Options: 'zscore', 'iqr', 'winsorize', 'lof'.
    :type method: str
    :param ignore_columns: The columns to ignore when handling outliers.
    :type ignore_columns: list
    :param contamination: The proportion of outliers in the data set. Default is 0.05.
    :type contamination: float
    :param threshold_sd: The minimum proportion of outliers in a sample (row) to consider for soft dropping. Default is 0.8.
    :type threshold_sd: float
    :param threshold_missing: The threshold for missing values. Default is 0.1.
    :type threshold_missing: float
    :param soft_drop: Boolean to indicate whether to softly drop outliers.
    :type soft_drop: bool
    :param threshold_corr: The correlation threshold to use. Default is 0.9.
    :type threshold_corr: float

    :return: The cleaned DataFrame.
    :rtype: pd.DataFrame
    """
    logger.info("Cleaning train and test data...")
    # format column types
    logger.info("Formatting column types...")
    train_df = format_dtype(train_df)
    test_df = format_dtype(test_df)

    # handle duplicates
    logger.info("Handling duplicates...")
    train_df.drop_duplicates(inplace=True)
    test_df.drop_duplicates(inplace=True)

    # remove outliers
    logger.info("Removing outliers...")
    cleaned_train_df = remove_outliers(train_df, method=method, ignore_columns=ignore_columns, contamination=contamination, threshold_sd=threshold_sd, soft_drop=soft_drop)
    cleaned_test_df = remove_outliers(test_df, method=method, ignore_columns=ignore_columns, contamination=contamination, threshold_sd=threshold_sd, soft_drop=soft_drop)

    logger.info("Filter features based train data...")
    # get features with missing values
    missing_values_features = identify_missing_values(train_df, threshold=threshold_missing)

    # get features with single unique values
    single_unique_features = identify_single_unique_features(cleaned_train_df)

    # get uncorrelated features with the target
    cleaned_train_df_with_target = calculate_RUL(cleaned_train_df, time_column='Cycle', group_column='UnitNumber')
    uncorrelated_features = get_uncorrelated_features(cleaned_train_df_with_target, threshold=threshold_corr, target='RUL')

    # make list of set of features to drop
    logger.info("Dropping features based on missing values, single unique values, and no target correlation...")
    ignore_set = set(ignore_columns)
    features_to_drop_set = {col for col in single_unique_features + uncorrelated_features + missing_values_features if
                            col not in ignore_set}
    features_to_drop = list(features_to_drop_set)

    # drop features
    cleaned_train_df.drop(columns=features_to_drop, inplace=True)
    cleaned_test_df.drop(columns=features_to_drop, inplace=True)

    logger.info("Data cleaning completed.")
    logger.info(f"Original train DataFrame shape: {train_df.shape}, Resulting train DataFrame shape: {cleaned_train_df.shape}")
    logger.info(f"Original test DataFrame shape: {test_df.shape}, Resulting test DataFrame shape: {cleaned_test_df.shape}")

    return cleaned_train_df, cleaned_test_df
