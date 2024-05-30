from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim.lr_scheduler import StepLR

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def scale_data(df_train, df_test):
    """
    Scales the numerical columns in the DataFrame using MinMaxScaler except "UnitNumber" and "RUL".
    
    Args:
        df_train (pandas.DataFrame): Training DataFrame.
        df_test (pandas.DataFrame): Testing DataFrame.
        
    Returns:
        pandas.DataFrame, pandas.DataFrame: Scaled training and testing DataFrames.
    """
    scaler = MinMaxScaler()
    
    # Separate "UnitNumber" and "RUL" columns
    unit_number_train = df_train[['UnitNumber']]
    unit_number_test = df_test[['UnitNumber']]
    rul_train = df_train[['RUL']]
    rul_test = df_test[['RUL']]
    
    # Select all columns except "UnitNumber" and "RUL"
    other_columns = df_train.columns.difference(['UnitNumber', 'RUL'])
    
    # Fit the scaler on the training data and transform it
    df_train_scaled = df_train.copy()
    df_train_scaled[other_columns] = scaler.fit_transform(df_train[other_columns])
    
    # Transform the test data using the same scaler
    df_test_scaled = df_test.copy()
    df_test_scaled[other_columns] = scaler.transform(df_test[other_columns])
    
    # Add "UnitNumber" and "RUL" back to the scaled data
    df_train_scaled = pd.concat([unit_number_train, rul_train, df_train_scaled[other_columns]], axis=1)
    df_test_scaled = pd.concat([unit_number_test, rul_test, df_test_scaled[other_columns]], axis=1)
    
    return df_train_scaled, df_test_scaled
    
def create_sliding_window_test(df, window_size=30, drop_columns=["UnitNumber", "RUL"], typ = "test"):
    """
    Creates a sliding window of data for time series prediction.

    Args:
        df (pandas.DataFrame): Input DataFrame containing time series data.
        window_size (int): Size of the sliding window.
        drop_columns (list): List of columns to drop from the input DataFrame.

    Returns:
        tuple: A tuple containing X (input) and y (output) arrays.
    """
    number_engines = df["UnitNumber"].unique()
    X, y = [], []

    for engine in number_engines:
        # Get data for the current engine
        temp = df[df["UnitNumber"] == engine]
        assert temp["UnitNumber"].unique() == engine

        X_temp = temp.iloc[-window_size:].drop(columns=drop_columns)
        
        # Ensure X_temp has the correct shape by padding if necessary
        if len(X_temp) < window_size:
            # Calculate the number of rows to pad
            pad_rows = window_size - len(X_temp)
            # Create a DataFrame of zeros with the same columns as X_temp
            padding = pd.DataFrame(0, index=np.arange(pad_rows), columns=X_temp.columns)
            # Concatenate the padding DataFrame with X_temp
            X_temp = pd.concat([padding, X_temp], ignore_index=True)
        
        # Ensure the final shape of X_temp
        assert X_temp.shape[0] == window_size
        
        # Convert X_temp to a NumPy array and append to the list
        X.append(X_temp.to_numpy())
        
        # Get the RUL value for the last cycle of the current engine
        Y_temp = temp.iloc[-1]["RUL"]
        y.append(Y_temp)
        
    # Convert lists to NumPy arrays
    X = np.array(X)
    y = np.array(y)

    return X, y
    
def create_sliding_window(df, window_size=30, drop_columns=["UnitNumber", "RUL"], typ = "train"):
    """
    Creates a sliding window of data for time series prediction.

    Args:
        df (pandas.DataFrame): Input DataFrame containing time series data.
        window_size (int): Size of the sliding window.
        drop_columns (list): List of columns to drop from the input DataFrame.

    Returns:
        tuple: A tuple containing X (input) and y (output) arrays.
    """
    if typ == "test":
        return create_sliding_window_test(df, window_size = window_size, drop_columns = drop_columns)
    else:
        number_engines = df["UnitNumber"].unique()
        X, y = [], []
    
        for engine in number_engines:
            # Get data for the current engine
            temp = df[df["UnitNumber"] == engine]
            assert temp["UnitNumber"].unique() == engine
    
            for i in range(len(temp) - window_size + 1):
                # Extract windowed data and RUL for each window
                X_temp = temp.iloc[i : (i + window_size)].drop(columns=drop_columns)
                Y_temp = temp.iloc[(i + window_size - 1)]["RUL"]
                assert len(X_temp) == window_size
                X.append(X_temp.to_numpy())
                y.append(Y_temp)
    
        X = np.array(X)
        y = np.array(y)
    
        return X, y


def calculate_RUL_test(test_data, RUL_data):
    RUL = []
    for i in RUL_data.iterrows():
        unit_num = i[0]
        val = i[1]["RUL"]
        tmp = test_data[test_data["UnitNumber"] == unit_num + 1]
        li = list(range(val + len(tmp) - 1, val - 1, -1))
        for j in li:
            RUL.append(j)
        assert RUL[-1] == val
    assert len(RUL) == len(test_data)
    test_data["RUL"] = RUL
    return test_data
    
