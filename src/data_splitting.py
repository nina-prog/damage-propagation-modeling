"""This file contains a collection of splitting functions that are used to split time series data into training and
validation sets."""
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from typing import Tuple

from src.logger import setup_logger

logger = setup_logger(__name__, level='INFO')  # Change the level to 'DEBUG' to see more information


def train_val_split_by_group(
        df: pd.DataFrame,
        group: str = "UnitNumber",
        test_size: float = 0.18,
        n_splits: int = 2,
        random_state: int = 7
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the DataFrame into training and validation sets based on a group identifier.

    Parameters:
    df (pd.DataFrame): The DataFrame to split.
    group (str): The column name to group by for splitting. Defaults to "UnitNumber".
    test_size (float): The proportion of the dataset to include in the test split. Defaults to 0.18.
    n_splits (int): Number of re-shuffling & splitting iterations. Defaults to 2.
    random_state (int): Random state for reproducibility. Defaults to 7.

    Returns:
    Tuple[pd.DataFrame, pd.DataFrame]: The training and test DataFrames.
    """
    # Initialize GroupShuffleSplit
    splitter = GroupShuffleSplit(test_size=test_size, n_splits=n_splits, random_state=random_state)

    # Perform the split
    split = splitter.split(df, groups=df[group])

    # Get the indices for the train and test sets
    train_inds, test_inds = next(split)

    # Create the train and test DataFrames using the indices
    train = df.iloc[train_inds]
    test = df.iloc[test_inds]

    # Log the number of unique groups and total rows in the train and test sets
    logger.info(f"Train set contains {train[group].nunique()} different engines --> in total {len(train)}")
    logger.info(f" Test set contains {test[group].nunique()} different engines --> in total {len(test)}")

    return train, test
