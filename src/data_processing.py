"""This file contains functions for the data processing"""
import numpy as np
import pandas as pd

from src.logger import setup_logger

logger = setup_logger(__name__, level='INFO')  # Change the level to 'DEBUG' to see more information


def apply_padding_on_train_data_and_test_data(train_data, test_data, window_size, padding: int = None,
                                              time_column: str = "Cycle", time_value_for_padding: int = -1,
                                              group_column: str = "UnitNumber"):
    """

    :param train_data: The training data.
    :type train_data: Pandas Dataframe
    :param test_data: The test data.
    :type test_data: Pandas Dataframe
    :param window_size: The size of the windows.
    :type window_size: int
    :param padding: The number of rows to pad to each motor. If padding None, it will be set to the minimum value,
        meaning that the minimum number of cycles of each motor is higher or equal to winndow_size.
    :type padding: int
    :param group_column: The name of the column that identifies the units.
    :type group_column: str
    :param time_value_for_padding: The value to use for the time column for the padded data.
    :type time_value_for_padding: int
    :param time_column: The name of the time column based on which the RUL will be calculated.
    :type time_column: str

    :return: The padded data, which has been sorted after padding by time_column and group_column
    :rtype: Tuple of two Pandas Dataframes
    """
    # Padding of train_data and test_data
    min_train = min([values.shape[0] for values in train_data.groupby(group_column).indices.values()])
    min_test = min([values.shape[0] for values in test_data.groupby(group_column).indices.values()])
    logger.info(f"The minimum number of cycles of a motor before the padding is {min(min_train, min_test)}.")

    if padding is None:
        padding = 0
        if window_size > min(min_train, min_test):
            padding = window_size - min(min_train, min_test)
        logger.info(f"The padding value is {padding}.")

    new_train_data = apply_padding_on_data(data=train_data, padding=padding, time_column=time_column,
                                           time_value_for_padding=time_value_for_padding, group_column=group_column)
    new_test_data = apply_padding_on_data(data=test_data, padding=padding, time_column=time_column,
                                          time_value_for_padding=time_value_for_padding, group_column=group_column)

    min_train = min([values.shape[0] for values in train_data.groupby(group_column).indices.values()])
    min_test = min([values.shape[0] for values in test_data.groupby(group_column).indices.values()])
    logger.info(f"The minimum number of cycles of a motor with the padding is {min(min_train, min_test)}.")
    return new_train_data, new_test_data


def apply_padding_on_data(data, padding, time_column: str = "Cycle", time_value_for_padding: int = -1,
                          group_column: str = "UnitNumber"):
    """

    :param data: The data.
    :type data: Pandas Dataframe
    :param padding: The number of rows to pad to each motor.
    :type padding: int
    :param group_column: The name of the column that identifies the units.
    :type group_column: str
    :param time_value_for_padding: The value to use for the time column for the padded data.
    :type time_value_for_padding: int
    :param time_column: The name of the time column based on which the RUL will be calculated.
    :type time_column: str

    :return: The padded data, which has been sorted after padding by time_column and group_column
    :rtype: Pandas Dataframe
    """
    new_data = data.copy()
    for engine_number in new_data[group_column].unique():
        padding_data_frame = pd.DataFrame(np.zeros(shape=(padding, len(new_data.columns))), columns=new_data.columns)
        padding_data_frame[group_column] = engine_number
        padding_data_frame[time_column] = time_value_for_padding
        new_data = pd.concat([new_data, padding_data_frame])
    new_data.sort_values([time_column, group_column], inplace=True, ascending=True)
    return new_data
