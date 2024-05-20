""" This module contains functions for outlier detection. """

import numpy as np
import pandas as pd
from typing import List, Union
from sklearn.neighbors import LocalOutlierFactor

from scipy.stats import zscore
from scipy.stats.mstats import winsorize

from src.logger import setup_logger

logger = setup_logger(__name__, level='DEBUG')  # Change the level to 'DEBUG' to see more information


def remove_outliers_zscore(df: pd.DataFrame, soft_drop: bool = False, threshold_sd: float = 0.5) -> pd.DataFrame:
    """Remove outliers using the Z-score method.

    :param df: The input DataFrame.
    :type df: pd.DataFrame
    :param soft_drop: Boolean to indicate whether to softly drop outliers.
    :type soft_drop: bool
    :param threshold_sd: The minimum proportion of outliers in a sample (row) to consider for soft dropping.
    :type threshold_sd: float

    :return: The DataFrame with outliers removed.
    :rtype: pd.DataFrame
    """
    z_scores = zscore(df)
    threshold = 3

    if soft_drop:
        outlier_proportion = (np.abs(z_scores) > threshold).mean(axis=1)
        num_samples_soft_dropped = outlier_proportion[outlier_proportion > threshold_sd].shape[0]
        logger.debug(f"Found {num_samples_soft_dropped} samples to be softly dropped.")
        result_df = df[(np.abs(z_scores) < threshold).all(axis=1) | (outlier_proportion <= threshold_sd)]
    else:
        num_samples_dropped = df[~(np.abs(z_scores) < threshold).all(axis=1)].shape[0]
        logger.debug(f"Found {num_samples_dropped} samples to be dropped.")
        result_df = df[(np.abs(z_scores) < threshold).all(axis=1)]

    logger.debug(f"Original DataFrame shape: {df.shape}, Resulting DataFrame shape: {result_df.shape}")
    return result_df


def remove_outliers_iqr(df: pd.DataFrame, threshold_iqr: float = 1.5, threshold_sd: float = 0.3 , soft_drop: bool = False) -> pd.DataFrame:
    """Remove outliers using the IQR method.

    :param threshold_sd: The minimum proportion of outliers in a sample (row) to consider for soft dropping.
    :type threshold_sd: float
    :param threshold_iqr: The threshold for the IQR method.
    :type threshold_iqr: float
    :param df: The input DataFrame.
    :type df: pd.DataFrame
    :param soft_drop: Boolean to indicate whether to softly drop outliers.
    :type soft_drop: bool

    :return: The DataFrame with outliers removed.
    :rtype: pd.DataFrame
    """
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - threshold_iqr * IQR
    upper_bound = Q3 + threshold_iqr * IQR

    if soft_drop:
        outlier_proportion = ((df < lower_bound) | (df > upper_bound)).mean(axis=1)
        num_samples_soft_dropped = outlier_proportion[outlier_proportion > threshold_sd].shape[0]
        logger.debug(f"Found {num_samples_soft_dropped} samples to be softly dropped.")
        result_df = df[((df < lower_bound) | (df > upper_bound)).all(axis=1) | (outlier_proportion <= threshold_sd)]
    else:
        num_samples_dropped = df[~((df < lower_bound) | (df > upper_bound)).all(axis=1)].shape[0]
        logger.debug(f"Found {num_samples_dropped} samples to be dropped.")
        result_df = df[((df < lower_bound) | (df > upper_bound)).all(axis=1)]

    logger.debug(f"Original DataFrame shape: {df.shape}, Resulting DataFrame shape: {result_df.shape}")
    return result_df


def remove_outliers_winsorize(df: pd.DataFrame, ignore_columns: List[str] = None, contamination: float = 0.05) -> pd.DataFrame:
    """Remove outliers using the Winsorization method. This method replaces the extreme values with the threshold value. The threshold value is determined by the proportion of outliers in the data set.

    :param df: The input DataFrame.
    :type df: pd.DataFrame
    :param ignore_columns: The columns to ignore when handling outliers.
    :type ignore_columns: list
    :param contamination: The proportion of outliers which are considered as such. Default is 0.05. If 0.05 then the upper and lower 5% of the data are considered as outliers. They are replaced by the 5th and 95th percentiles respectively.
    :type contamination: float

    :return: The DataFrame with outliers removed.
    :rtype: pd.DataFrame
    """
    if ignore_columns is None:
        ignore_columns = []

    num_outliers = int(df.shape[0] * contamination)
    logger.debug(f"Found {num_outliers} outliers to be replaced (winsorized).")
    result_df = df.apply(lambda x: winsorize(x, limits=[contamination, contamination]) if x.name not in ignore_columns else x)
    logger.debug(f"Original DataFrame shape: {df.shape}, Resulting DataFrame shape: {result_df.shape}")

    return result_df


def remove_outliers_with_lof(df: pd.DataFrame, contamination: float = 0.05) -> pd.DataFrame:
    """Handle outliers using the Local Outlier Factor (LOF) method. This method calculates the local density around each data point and identifies outliers as points with significantly lower densities compared to their neighbors.

    :param df: The input DataFrame.
    :type df: pd.DataFrame
    :param contamination: The proportion of outliers in the data set. Default is 0.05.
    :type contamination: float

    :return: The DataFrame with outliers removed.
    :rtype: pd.DataFrame
    """
    # fit the LOF model
    lof = LocalOutlierFactor(contamination=contamination, novelty=False)
    yhat = lof.fit_predict(df)
    lof_scores = -lof.negative_outlier_factor_

    result_df = df[yhat != -1]
    logger.debug(f"Found {df.shape[0] - result_df.shape[0]} outliers to be dropped.")
    logger.debug(f"Original DataFrame shape: {df.shape}, Resulting DataFrame shape: {result_df.shape}")

    return result_df


def remove_outliers(df: pd.DataFrame, method: Union[str, None] = 'winsorize', ignore_columns: List[str] = None, contamination: float = 0.05, threshold_sd: float = 0.8, soft_drop: bool = False) -> pd.DataFrame:
    """Remove outliers from the input DataFrame using the specified method.

    :param df: The input DataFrame.
    :type df: pd.DataFrame
    :param method: The outlier detection method to use. Options: 'zscore', 'iqr', 'winsorize', 'lof', 'elliptic'.
    :type method: str
    :param ignore_columns: The columns to ignore when handling outliers.
    :type ignore_columns: list
    :param contamination: The proportion of outliers in the data set. Default is 0.05.
    :type contamination: float
    :param threshold_sd: The minimum proportion of outliers in a sample (row) to consider for soft dropping. Default is 0.8.
    :type threshold_sd: float
    :param soft_drop: Boolean to indicate whether to softly drop outliers.
    :type soft_drop: bool

    :return: The DataFrame with outliers removed.
    :rtype: pd.DataFrame
    """
    if ignore_columns is None:
        ignore_columns = []

    logger.debug(f"Removing outliers using method: {method} ...")

    match method:
        case 'zscore':
            result_df = remove_outliers_zscore(df, soft_drop=soft_drop, threshold_sd=threshold_sd)
        case 'iqr':
            result_df = remove_outliers_iqr(df, soft_drop=soft_drop, threshold_sd=threshold_sd)
        case 'winsorize':
            result_df = remove_outliers_winsorize(df, ignore_columns=ignore_columns, contamination=contamination)
        case 'lof':
            result_df = remove_outliers_with_lof(df, contamination=contamination)
        case None:
            logger.info("No outlier detection method specified. Skipping outlier detection.")
            result_df = df
        case _:
            raise ValueError(f"Invalid method: {method}. Please choose from 'zscore', 'iqr', 'winsorize', 'lof'.")

    return result_df
