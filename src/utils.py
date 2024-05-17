"""This file contains a collection of utility functions that can be used for common tasks in this project."""
import pandas as pd
import yaml

from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, r2_score

from typing import Any, Dict, Generator, Tuple

from src.logger import setup_logger

logger = setup_logger(__name__, level='INFO')  # Change the level to 'DEBUG' to see more information


def flatten(nested_list):
    """Flatten a nested list.

    :param nested_list: The nested list to flatten.
    :type nested_list: list

    :return: The flattened list.
    :rtype: list
    """
    return [item for sublist in nested_list for item in sublist]


def load_config(config_path: str) -> dict:
    """
    Loads a YAML configuration file.

    :param config_path: Path to the configuration file
    :type config_path: str

    :return: Configuration dictionary
    :rtype: dict
    """
    try:
        with open(config_path, "r") as ymlfile:
            return yaml.load(ymlfile, yaml.FullLoader)
    except FileNotFoundError:
        raise FileNotFoundError(f"File {config_path} not found!")
    except PermissionError:
        raise PermissionError(f"Insufficient permission to read {config_path}!")
    except IsADirectoryError:
        raise IsADirectoryError(f"{config_path} is a directory!")


def load_data(config_path: str, dataset_num: int) -> tuple:
    """Load the specified dataset.

    :param dataset_num: The number of the dataset to load.
    :type dataset_num: int
    :param config_path: The path to the configuration file.
    :type config_path: str

    :return: The loaded data.
    :rtype: tuple
    """
    logger.info(f"Loading data set {dataset_num}...")

    # Load the configurations
    config = load_config(config_path)

    # Access data sets information
    data_sets = config.get('dataloading', {}).get('sets', [])
    if not data_sets:
        raise ValueError("No datasets found in the configuration.")
    data_dir = config.get('dataloading', {}).get('data_dir', '')
    if not data_dir:
        raise ValueError("Data directory not found in the configuration.")

    # Check if given dataset number is valid
    if not 1 <= dataset_num <= len(data_sets):
        raise ValueError(f"Dataset number must be between 1 and {len(data_sets)}.")

    # Access the paths to the selected data
    selected_set = data_sets[dataset_num - 1]
    train_path = data_dir + selected_set.get('train', '')
    test_path = data_dir + selected_set.get('test', '')
    test_RUL_path = data_dir + selected_set.get('RUL', '')

    # Load the data
    column_names = flatten(config.get('dataloading', {}).get('columns', []))
    train_data = pd.read_csv(train_path, header=None, names=column_names, sep=r'\s+', decimal=".")
    test_data = pd.read_csv(test_path, header=None, names=column_names, sep=r'\s+', decimal=".")
    test_RUL_data = pd.read_csv(test_RUL_path, header=None, names=['RUL'], sep=r'\s+', decimal=".")

    logger.info(f"Loaded raw data for dataset {dataset_num}.")
    logger.info(f"Train Data: {train_data.shape}")
    logger.info(f"Test Data: {test_data.shape}")
    logger.info(f"Test RUL Data: {test_RUL_data.shape}")

    return train_data, test_data, test_RUL_data


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

# Example usage:
# for train_df, val_df in k_fold_group_cross_validation(df):
#     # train your model on train_df
#     # validate your model on val_df
def train_and_evaluate_model(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    n_splits: int = 5,
) -> Dict[str, list]:
    """
    Train and evaluate a model using the specified cross-validation strategy.

    Parameters:
    model (Any): The model to be trained and evaluated.
    X (pd.DataFrame): The feature matrix.
    y (pd.Series): The target variable.
    groups (pd.Series): The group labels for cross-validation.
    cv (Generator): Cross-validation strategy.
    scoring (Dict[str, make_scorer]): The scoring metrics.

    Returns:
    Dict[str, list]: Cross-validation scores for each defined metric.
    """
    cv = GroupKFold(n_splits=n_splits)
    # Define the scoring metrics for regression
    scoring: Dict[str, make_scorer] = {
        'mae': make_scorer(mean_absolute_error),
        'mse': make_scorer(mean_squared_error),
        'r2': make_scorer(r2_score)
    }

    # Perform cross-validation
    scores = cross_validate(model, X, y, cv=cv, groups=groups, scoring=scoring, return_train_score=False)
    
    # Log the results
    for metric in scoring.keys():
        logger.info(f"{metric.upper()} Scores: {scores['test_' + metric]}")
        logger.info(f"Average {metric.upper()}: {scores['test_' + metric].mean():.4f}")
    
    return scores
