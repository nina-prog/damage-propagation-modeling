"""This file contains functions for model evaluation."""
import pandas as pd
from sklearn.model_selection import GroupKFold
from typing import Any, Dict, Generator, Tuple

from src.logger import setup_logger

logger = setup_logger(__name__, level='INFO')  # Change the level to 'DEBUG' to see more information


def k_fold_group_cross_validation(
        df: pd.DataFrame,
        group: str = "UnitNumber",
        n_splits: int = 5
) -> Generator[Tuple[pd.DataFrame, pd.DataFrame], None, None]:
    """
    Performs K-fold group cross-validation.

    Parameters:
    df (pd.DataFrame): The DataFrame to split.
    group (str): The column name to group by for splitting. Defaults to "UnitNumber".
    n_splits (int): Number of folds. Defaults to 5.
    random_state (int): Random state for reproducibility. Defaults to None.

    Yields:
    Generator[Tuple[pd.DataFrame, pd.DataFrame], None, None]:
        A generator yielding tuples of (train DataFrame, validation DataFrame) for each fold.
    """
    # Initialize GroupKFold
    group_kfold = GroupKFold(n_splits=n_splits)

    # Iterate over each fold
    for fold, (train_inds, val_inds) in enumerate(group_kfold.split(df, groups=df[group])):
        # Create the train and validation DataFrames using the indices
        train = df.iloc[train_inds]
        val = df.iloc[val_inds]

        # Log the number of unique groups and total rows in the train and validation sets
        logger.info(f"Fold {fold + 1}:")
        logger.info(f"Train set contains {train[group].nunique()} different engines --> in total {len(train)}")
        logger.info(f"Validation set contains {val[group].nunique()} different engines --> in total {len(val)}")

        yield train, val
