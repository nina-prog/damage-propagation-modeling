"""This file contains a collection of utility functions that can be used for common tasks in this project."""
import pandas as pd
import yaml
import os
from src.data_preprocessing import calculate_RUL

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


def load_data(config_path: str, dataset_num: int, raw=False) -> tuple:
    """Load the specified dataset.

    :param config_path: The path to the configuration file.
    :type config_path: str
    :param path_to_data: The path to the data.
    :type path_to_data: str
    :param raw: Whether to return the raw data or combined feature, target data.
    :type raw: bool

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
    RUL_path = data_dir + selected_set.get('RUL', '')

    # Load the data
    column_names = flatten(config.get('dataloading', {}).get('columns', []))
    train_data = pd.read_csv(train_path, header=None, names=column_names, sep=r'\s+', decimal=".")
    test_data = pd.read_csv(test_path, header=None, names=column_names, sep=r'\s+', decimal=".")
    RUL_data = pd.read_csv(RUL_path, header=None, names=['RUL'], sep=r'\s+', decimal=".")

    if raw:
        logger.info(f"Loaded raw data for dataset {dataset_num}.")
        logger.info(f"Train Data: {train_data.shape}")
        logger.info(f"Test Data: {test_data.shape}")
        logger.info(f"RUL Data: {RUL_data.shape}")
        return train_data, test_data, RUL_data

    # Map the RUL data to the test data it belongs to
    test_data['RUL'] = RUL_data['RUL']

    # Calculate the RUL for the training data
    train_data = calculate_RUL(data=train_data, time_column="Cycle", group_column="UnitNumber")

    logger.info(f"Loaded feature and target data for dataset {dataset_num}.")
    logger.info(f"Train Data: {train_data.shape}")
    logger.info(f"Test Data: {test_data.shape}")
    return train_data, test_data
