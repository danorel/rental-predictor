from datetime import date
from decimal import Decimal
from pathlib import Path
import torch
from sklearn.metrics import mean_squared_error


def alchemyencoder(obj):
    """JSON encoder function for SQLAlchemy special classes."""
    if isinstance(obj, date):
        return obj.isoformat()
    elif isinstance(obj, Decimal):
        return float(obj)


def download_dataset_from(
        url_to_db,
        path_to_csv,
        csv_filename
):
    """
    Upload db into csv file, located in path_to_csv.
    Connect to postgres database by the url_to_db.
    :type csv_filename: str
    :type path_to_csv: str
    :type url_to_db: str
    """
    status, is_downloaded = 'OK', False
    path = Path(path_to_csv)
    if not path.exists():
        path.mkdir(parents=True)
    if not (path / csv_filename).exists():
        from project.app.queries import get_dataset
        df = get_dataset(
            url=url_to_db
        )
        df.to_csv(path_to_csv)
        is_downloaded = True
    return status, is_downloaded


# def choose_model(
#         model,
#         available_models=['nn', 'xgboost', 'decision_tree']
# ):
#     """
#     Choose model due to the input parameter
#     :param model: str
#     :param available_models: list
#     """
#     if model == "nn":
#         return nn.NeuralNetwork
#     elif model == "xgboost":
#         return decision_tree.DecisionTreeRegressor
#     elif model == "decision_tree":
#         return xgboost.XGBRegressor
#     else:
#         raise AttributeError(
#             f'Fatal error. Unavailable model chosen. Choose from: {available_models}'
#         )


def load_features() -> dict:
    from features import NUMERICAL_FEATURES, BOOLEAN_FEATURES, CATEGORIAL_FEATURES, USELESS_FEATURES, SUBSTITUTE_FEATURES, TARGET
    features = {
        'numerical_features': NUMERICAL_FEATURES,
        'boolean_features': BOOLEAN_FEATURES,
        'categorial_features': CATEGORIAL_FEATURES,
        'useless_features': USELESS_FEATURES,
        'substitute_features': SUBSTITUTE_FEATURES,
        'target': TARGET
    }
    return features


def train_selected_model(
        selected_model,
        path_to_csv,
        csv_filename,
        features,
        config
) -> tuple:
    """
    Train the chosen model
    :type selected_model: str
    :type path_to_csv: str
    :type csv_filename: str
    :type features: dict
    :type config: dict
    """
    path = Path(path_to_csv) / csv_filename
    if selected_model == "nn":
        import project.labaratory.models.nn as network
        from project.labaratory.models.nn import model_selection
        from project.labaratory.utilities import load_dataset, prepare_data
        X, y = load_dataset(
            csv=path,
            numerical_features=features.get('numerical_features'),
            boolean_features=features.get('boolean_features'),
            categorial_features=features.get('categorial_features'),
            substitute_features=features.get('substitute_features'),
            useless_features=features.get('useless_features'),
            target=features.get('target')
        )
        print(f"X shapes: {X.shape}")
        print(f"y shapes: {y.shape}")
        train_loader, test_loader, validation_loader = prepare_data(
            X=X,
            y=y,
            batch_size=config.get('batch_size'),
            valid_size=config.get('validation_size'),
            test_size=config.get('test_size')
        )
        model = network.NeuralNetwork(
            input_dimension=X.shape[1],
            hidden_dimension=config.get('nn').get('hidden_dimension'),
            output_dimension=1,
            activation=config.get('nn').get('activation')
        )
        optimizer = torch.optim.SGD(
            params=model.parameters(),
            lr=config.get('params').get('lr')
        )
        criterion = torch.nn.MSELoss()
        metric = mean_squared_error

        weights, train_statistics = model_selection.train_model(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            train_loader=train_loader,
            num_epochs=config.get('num_epochs'),
            metric=metric,
            device=config.get('device'),
            print_step=config.get('print_step')
        )
        print(train_statistics)
        validation_predictions, validation_metrics = model_selection.test_model(
            model=model,
            loader=validation_loader,
            metric=metric,
            device=config.get('device'),
            isValidation=True
        )
        test_predictions, test_metrics = model_selection.test_model(
            model=model,
            loader=test_loader,
            metric=metric,
            device=config.get('device'),
            isValidation=False
        )
        return weights, train_statistics, validation_predictions, validation_metrics, test_predictions, test_metrics


# def predict(
#         model,
#         parameters,
#         path_to_csv,
#         csv_filename
# ):
#     X, y = utilities.load_dataset(
#         csv=path,
#         target=TARGET,
#         categorial_features=CATEGORIAL_FEATURES,
#         boolean_features=BOOLEAN_FEATURES,
#         numerical_features=NUMERICAL_FEATURES,
#         useless_features=USELESS_FEATURES,
#         substitute_features=SUBSTITUTE_FEATURES
#     )
#     prediction = None
#     return prediction
