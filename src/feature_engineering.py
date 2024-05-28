""" This module contains functions for custom feature engineering. """
import numpy as np
from scipy.signal import find_peaks
from typing import Union, Dict
import pandas as pd

from src.logger import setup_logger

logger = setup_logger(__name__, level='DEBUG')  # Change the level to 'DEBUG' to see more information


def extract_num_of_peaks_from_sensor_signals(dataframe, sensor_columns_prefix, add_as_new_feature=False) -> Union[pd.DataFrame, Dict[str, int]]:
    """
    Extract the absolut number of peaks for each sensor signal. The function returns the number of peaks for each sensor
    signal as a dictionary or adds the number of peaks as a new feature to the dataframe.

    :param dataframe: The input dataframe with the sensor signals.
    :type dataframe: pandas.DataFrame
    :param sensor_columns_prefix: The prefix of name of the columns that contain the sensor signals.
    :type sensor_columns_prefix: str
    :param: add_as_new_feature: Boolean to indicate whether to add the number of peaks as a new feature to the dataframe. If True, the function returns the dataframe with the new features.
    :type add_as_new_feature: bool

    :return: The dataframe with the new features or a dictionary with the number of peaks for each sensor signal.
    :rtype: pandas.DataFrame or dict
    """
    sensor_measure_columns_names = [column_name for column_name in dataframe.columns if column_name.startswith(sensor_columns_prefix)]

    absolut_count_of_peaks = {}
    for sensor_measure_column_name in sensor_measure_columns_names:
        values = dataframe[sensor_measure_column_name].values
        peaks, _ = find_peaks(values)
        num_of_peaks = len(peaks)
        absolut_count_of_peaks[sensor_measure_column_name] = num_of_peaks
        logger.debug(f"Absolut number of peaks for {sensor_measure_column_name}: {num_of_peaks}")

    if add_as_new_feature:
        df = dataframe.copy()
        for sensor_measure_column_name in sensor_measure_columns_names:
            df[sensor_measure_column_name + "__abs_num_peaks"] = absolut_count_of_peaks[sensor_measure_column_name]
        return df
    else:
        return absolut_count_of_peaks
