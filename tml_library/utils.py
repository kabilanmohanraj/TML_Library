import pandas as pd
import logging
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import json

from tml_library.feature_selection import FeatureSelector

def setup_logger(name, log_file):
    logger = logging.getLogger(name)
    handler = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def initialize_train_pipeline(config_path="config.json"):
    # Load configuration
    with open(config_path, "r") as f:
        config = json.load(f)

    # Load data with no headers, then transpose and split
    data = pd.read_csv(config["data_path"], header=None).T
    predictor_variable = data.iloc[:, 0]
    X = data.iloc[:, 1:]
    y = predictor_variable

    # Set up logger
    logger = setup_logger("pipeline_logger", config["log_file"])

    # Initialize model and hyperparameters based on the model name
    model_name = config["model_name"]
    if model_name == "SVC":
        model = SVC()
        # Define SVC-specific hyperparameter grid
        param_grid = {
            'C': config["model_params"].get("C", [0.1, 1, 5, 10, 30]),
            'kernel': config["model_params"].get("kernel", ['linear'])
        }
    elif model_name == "RandomForest":
        model = RandomForestClassifier()
        # Define RandomForest-specific hyperparameter grid
        param_grid = {
            'n_estimators': config["model_params"].get("n_estimators", [50, 100, 200, 500]),
            'max_depth': config["model_params"].get("max_depth", [None, 10, 20]),
            'min_samples_split': config["model_params"].get("min_samples_split", [2, 5, 10])
        }
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    return X, y, model, param_grid, logger


def initialize_test_pipeline(config_path="config.json"):
    # Load configuration
    with open(config_path, "r") as f:
        config = json.load(f)

    # Load test data
    data = pd.read_csv(config["test_data_path"])
    X_test = data.drop(config["target_column"], axis=1)
    y_test = data[config["target_column"]]

    # Set up logger
    logger = setup_logger("test_logger", config["log_file"])

    # Load the trained model
    model = load(config["saved_model_path"])

    return X_test, y_test, model, logger
