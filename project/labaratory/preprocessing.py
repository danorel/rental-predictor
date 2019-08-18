import pandas as pd
import ast

from sklearn.preprocessing import StandardScaler


def scale_features(
        df,
        numeric_features
) -> pd.DataFrame:
    """
    Scale the numerical features and return the pandas data frame with that modifications
    :type df: pd.DataFrame
    :type numeric_features: list
    """
    scaled_features = df[numeric_features]
    scaled_features = StandardScaler() \
        .fit_transform(scaled_features)
    df[numeric_features] = scaled_features
    return df


def substitute(
        df,
        substitute_features
) -> pd.DataFrame:
    """
    Substitute features with len property
    :type df: pd.DataFrame
    :type substitute_features: list
    """
    for feature in substitute_features:
        df[feature] = df[feature].map(lambda value: feature_to_len(feature, value))
    return df


def feature_to_len(
        feature,
        value
):
    """
    Extract the length of the feature
    :type feature: str
    :type value: object
    """
    if not isinstance(value, object) or pd.isna(value) or pd.isnull(value):
        return 0
    if feature == 'description':
        return len(str(value))
    if feature == 'image_urls':
        value = str(value)
        value = ast.literal_eval(value)
        return len(value)
    return 0
