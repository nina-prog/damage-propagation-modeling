"""This file contains functions for model evaluation."""
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, r2_score

from typing import Any, Dict, Generator, Tuple

from src.logger import setup_logger

logger = setup_logger(__name__, level='INFO')  # Change the level to 'DEBUG' to see more information


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
