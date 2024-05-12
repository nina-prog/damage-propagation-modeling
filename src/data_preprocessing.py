"""This file contains a collection of utility functions that can be used for common tasks in this project."""
import pandas as pd
from tsfresh.feature_extraction import extract_features
from tsfresh.feature_extraction.settings import MinimalFCParameters
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.utilities.dataframe_functions import roll_time_series

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
    data['RUL'] = data.groupby(group_column)[time_column].transform("max") - data[time_column]
    # Adding one because the current cycle also counts (RUL at failure is 1)
    data['RUL'] = data['RUL'] + 1
    logger.debug("RUL generated successfully.")

    return data


def create_rolling_windows_datasets(train_data: pd.DataFrame, test_data: pd.DataFrame, test_RUL_data: pd.DataFrame, column_id: str = "UnitNumber", column_sort: str = "Cycle", max_timeshift: int = 20, min_timeshift: int = 5) -> tuple:
    """Create rolling windows datasets for train and test data.

    :param train_data: The training data.
    :type train_data: pd.DataFrame
    :param test_data: The test data.
    :type test_data: pd.DataFrame
    :param test_RUL_data: The RUL data for the test data.
    :type test_RUL_data: pd.DataFrame
    :param column_id: The column name that identifies the units.
    :type column_id: str
    :param column_sort: The column name that sorts the data.
    :type column_sort: str
    :param max_timeshift: The maximum number of cycles of a rolling window.
    :type max_timeshift: int
    :param min_timeshift: The minimum number of cycles of a rolling window.
    :type min_timeshift: int

    :return: The train and test datasets.
    :rtype: tuple
    """

    # train data
    # generate rolling windows for train data
    train_data_rolled = roll_time_series(train_data, column_id=column_id, column_sort=column_sort, max_timeshift=max_timeshift, min_timeshift=min_timeshift)
    # extract features for train data
    X_train = extract_features(train_data_rolled.drop([column_id], axis=1),
                               column_id="id", column_sort=column_sort,
                               default_fc_parameters=MinimalFCParameters(),
                               impute_function=impute, show_warnings=False)
    # add index names
    X_train.index = X_train.index.rename([column_id, column_sort])
    # calculate target
    train_data = calculate_RUL(data=train_data, time_column=column_sort, group_column=column_id)
    # extract target
    y_train = train_data.set_index([column_id, column_sort]).sort_index().RUL.shift(-1)
    # make X and y consistent
    y_train = y_train[y_train.index.isin(X_train.index)]
    X_train = X_train[X_train.index.isin(y_train.index)]

    # test data
    # generate rolling windows for test data
    test_data_rolled = roll_time_series(test_data, column_id=column_id, column_sort=column_sort, max_timeshift=max_timeshift, min_timeshift=min_timeshift)
    # filter to only include the last window of each unit
    filtered_test_data_rolled = test_data_rolled.groupby(column_id).tail(max_timeshift)
    # extract features for test data
    X_test = extract_features(filtered_test_data_rolled.drop([column_id], axis=1),
                              column_id="id", column_sort=column_sort,
                              default_fc_parameters=MinimalFCParameters(),
                              impute_function=impute, show_warnings=False)
    # add index names
    X_test.index = X_test.index.rename([column_id, column_sort])
    # extract target for test data, match index of y_test with X_test
    y_test = test_RUL_data
    y_test.index = X_test.index

    return X_train, y_train, X_test, y_test
