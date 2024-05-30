"""This file contains a collection of utility functions that can be used for common tasks in this project."""
import pandas as pd
import yaml

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
