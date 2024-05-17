""" This module contains functions for data cleaning. """
import pandas as pd
from typing import Union, List

from src.logger import setup_logger

logger = setup_logger(__name__, level='INFO')  # Change the level to 'DEBUG' to see more information


def identify_missing_values(df: pd.DataFrame, threshold: float = 0.1, drop: bool = False) -> Union[pd.DataFrame, List[str]]:
    """
    Identify features with missing values above a certain threshold.

    :param df: DataFrame with features to investigate.
    :param threshold: Threshold for the relative amount of missing values.
    :param drop: Boolean to indicate whether to drop the features from the DataFrame.
    :return: DataFrame or list
    """
    # Calculate the total number of missing values for each feature
    missing_values = df.isnull().sum()
    total_rows = len(df)
    # Calculate the relative amount of missing values for each feature
    relative_missing_values = missing_values / total_rows
    # Identify feature names where the relative amount is above the threshold
    features_above_threshold = relative_missing_values[relative_missing_values > threshold].index.tolist()

    logger.info(
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

    logger.info(f"Found {len(single_unique_features)} features with only a single unique value.")

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

    logger.info(f"Found {len(cat_cols)} categorical columns: {cat_cols}")

    return df


# TODO: Implement match case for Python 3.10
# TODO: Import the necessary functions from the required libraries
