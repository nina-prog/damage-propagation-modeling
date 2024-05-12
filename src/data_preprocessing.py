"""This file contains a collection of utility functions that can be used for common tasks in this project."""
import pandas as pd

from src.logger import setup_logger

logger = setup_logger(__name__, level='INFO')  # Change the level to 'DEBUG' to see more information


def calculate_RUL(data: pd.DataFrame, time_column: str, group_column: str) -> pd.DataFrame:
    """Generate the remaining useful life (RUL) for the dataset. The RUL is the number of cycles before the machine
    fails. RUL at failure is 1.

    :param time_column: The name of the time column based on which the RUL will be calculated.
    :type time_column: str
    :param data: The dataset.
    :type data: pd.DataFrame

    :return: The dataset with the RUL column.
    :rtype: pd.DataFrame
    """

    data = data.copy()
    data_max = pd.DataFrame(data[group_column].unique(), columns=[group_column])
    # Note: You have to get the numpy array, otherwise pandas will map the data incorrectly.
    # In our case it will take the UnitNumber column as index, and it starts at 1 and not at 0 thus the first value
    # would be NaN.
    data_max['MAX'] = data.groupby(group_column)[time_column].max().values

    data = pd.merge(data, data_max, on=group_column, how='left')
    data['RUL'] = data['MAX'] - data[time_column]
    data = data.drop('MAX', axis=1)
    logger.debug("RUL generated successfully.")

    return data
