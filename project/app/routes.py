import json

from flask import request, jsonify

from project.app import db, app
from project.app.queries import get_statistics, get_records, get_dataset
from project.app.utilities import download_dataset_from, train_selected_model, load_features


@app.route('/api/v1/statistics')
def show_statistics():
    status, result = get_statistics(
        db=db,
        url=app.config.get('SQLALCHEMY_DATABASE_URI')
    )
    return jsonify({
        'status': status,
        'result': result
    })


@app.route('/api/v1/records')
def show_records():
    limit = request.args.get('limit')
    offset = request.args.get('offset')
    status, result = get_records(
        db=db,
        url=app.config.get('SQLALCHEMY_DATABASE_URI'),
        limit=limit,
        offset=offset
    )
    return jsonify({
        'status': status,
        'result': result
    })


@app.route('/api/v1/download/dataset')
def download_dataset():
    status, is_downloaded = download_dataset_from(
        url_to_db=app.config.get('SQLALCHEMY_DATABASE_URI'),
        path_to_csv=app.config.get('PATH_TO_CSV'),
        csv_filename=app.config.get('CSV_FILENAME'),
    )
    return jsonify({
        "status": status,
        "is_downloaded": is_downloaded
    })


@app.route('/api/v1/train/model')
def train_model():
    selected_model = request.args.get('model')
    print(app.config.get('PATH_TO_CONFIGURATIONS'))
    with open(app.config.get('PATH_TO_CONFIGURATIONS'), 'r') as file:
        config = json.load(file)
    print(config)
    features = load_features()
    weights, training_stats, validation_predictions, validation_metrics, test_predictions, test_metrics = train_selected_model(
        selected_model=selected_model,
        path_to_csv=app.config.get('PATH_TO_CSV'),
        csv_filename=app.config.get('CSV_FILENAME'),
        config=config,
        features=features
    )
    return jsonify({
        "training": {
            "weights": weights,
            "stats": training_stats,
        },
        "validation": {
            "predictions": validation_predictions,
            "metrics": validation_metrics
        },
        "testing": {
            "predictions": test_predictions,
            "metrics": test_metrics
        }
    })

#
# @app.route('/api/v1/price/predict')
# def make_prediction():
#     model = choose_model(
#         model=request.args.get('model')
#     )
#     params = request.args.get('features')
#     prediction = predict(
#         model=model,
#         parameters=params,
#     )
#     return jsonify({
#         "prediction": prediction
#     })
