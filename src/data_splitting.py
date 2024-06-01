"""This file contains a collection of splitting functions that are used to split time series data into training and
validation sets."""
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from typing import Tuple, Union

from src.logger import setup_logger

logger = setup_logger(__name__, level='INFO')  # Change the level to 'DEBUG' to see more information


def train_val_split_by_group(
        X: pd.DataFrame,
        y: pd.DataFrame,
        group: str = "UnitNumber",
        test_size: float = 0.18,
        n_splits: int = 2,
        random_state: int = 7
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits the DataFrame into training and validation sets based on a group identifier. It makes sure that all the data
    for a specific group is either in the training set or in the validation set.

    :param X: The features DataFrame.
    :type X: pd.DataFrame
    :param y: The target DataFrame.
    :type y: pd.DataFrame
    :param group: The name of the column or the index to use for grouping.
    :type group: str
    :param test_size: The proportion of the dataset to include in the validation set.
    :type test_size: float
    :param n_splits: The number of re-shuffling & splitting iterations.
    :type n_splits: int
    :param random_state: The seed used by the random number generator.
    :type random_state: int

    :return: The X_train, X_val, y_train, and y_val DataFrames.
    :rtype: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
    """
    index = None
    if group in X.index.names:
        index = list(X.index.names)
        X = X.reset_index()
        y = y.reset_index()
    elif group not in X.columns:
        raise ValueError(f"Group identifier {group} not found in the DataFrame.")

    splitter = GroupShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
    split = splitter.split(X, y, groups=X[group])

    train_idx, val_idx = next(split)
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    num_unique_groups_tr = len(X_train[group].unique())
    num_unique_groups_val = len(X_val[group].unique())

    if index is not None:
        X_train.set_index(index, inplace=True)
        X_val.set_index(index, inplace=True)
        y_train.set_index(index, inplace=True)
        y_val.set_index(index, inplace=True)

    logger.info("Split data successfully.")
    logger.info(f"Train set contains {num_unique_groups_tr} different engines --> in total {len(X_train)}")
    logger.info(f"Validation set contains {num_unique_groups_val} different engines --> in total {len(X_val)}")

    return X_train, X_val, y_train, y_val
