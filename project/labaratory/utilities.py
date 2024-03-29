import numpy as np
import pandas as pd
import torch

from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

from project.labaratory.preprocessing import scale_features, substitute


def load_dataset(
        csv,
        target,
        categorial_features,
        boolean_features,
        numerical_features,
        useless_features,
        substitute_features
) -> tuple:
    """
    Load the dataset, preprocess it, split it in X, y numpy variables
    :type csv: str
    :type target: str
    :type categorial_features: list
    :type boolean_features: list
    :type numerical_features: list
    :type useless_features: list
    :type substitute_features: list
    """
    print("Collecting the data from csv file...")
    df = pd.read_csv(csv)
    print("Substituting the features...")
    df = substitute(
        df=df,
        substitute_features=substitute_features
    )
    print("Dropping the useless features...")
    df.drop(
        columns=useless_features,
        inplace=True
    )
    print("Filling the NA values...")
    df.fillna(
        value=df.mean(),
        inplace=True
    )
    print("Scaling the numeric features...")
    df = scale_features(
        df=df,
        numeric_features=numerical_features
    )
    print("Performing one-hot-encoding...")
    df = pd.get_dummies(
        data=df,
        columns=(categorial_features + boolean_features),
    )
    print("Extracting the X, y features from the pandas DataFrame object...")
    X, y = extract_X_y(
        df=df,
        target=target
    )
    return X, y


def extract_X_y(
        df,
        target
):
    """
    Get numpy arrays from pandas data frame
    :type df: pd.DataFrame
    :type target: str
    """
    return np.array(df.loc[:, df.columns != target]), np.array(df.loc[:, df.columns == target])


def prepare_data(
        X,
        y,
        batch_size,
        test_size=.2,
        valid_size=.1,
        random_state=42
):
    """
    Creating the data loaders
    :type X: np.ndarray
    :type y: np.ndarray
    :type batch_size: int
    :type test_size: float
    :type valid_size: float
    :type random_state: int
    """
    print("Splitting the train numpy array into train/test numpy arrays...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )
    print("Splitting the train numpy array into train/validation numpy arrays...")
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train,
        y_train,
        test_size=valid_size,
        random_state=random_state
    )
    print("Preparing the train dataset...")
    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    print("Preparing the validation dataset...")
    valid_dataset = TensorDataset(torch.from_numpy(X_valid), torch.from_numpy(y_valid))
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    print("Preparing the test dataset...")
    test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    return train_loader, test_loader, valid_loader
