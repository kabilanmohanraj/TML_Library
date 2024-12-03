import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import json
import logging
import pickle

from tml_library.feature_selection import FeatureSelector

from gi.repository import GLib


# Custom logging handler for TextView
class TextViewHandler(logging.Handler):
    def __init__(self, textview):
        super().__init__()
        self.textview = textview

    def emit(self, record):
        log_entry = self.format(record)
        GLib.idle_add(self.append_text, log_entry)

    def append_text(self, log_entry):
        buffer = self.textview.get_buffer()
        buffer.insert(buffer.get_end_iter(), log_entry + '\n')

# Logger setup function
def setup_logger(log_file, textview):
    logger = logging.getLogger('TMLPipelineLogger')
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s')
        file_handler.setFormatter(formatter)

        # TextView handler
        textview_handler = TextViewHandler(textview)
        textview_handler.setLevel(logging.INFO)
        textview_handler.setFormatter(formatter)

        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(textview_handler)

    return logger


def initialize_train_pipeline(config_path="config.json", test_size=0.2, random_state=42):
    # Load configuration
    with open(config_path, "r") as f:
        config = json.load(f)

    # Load data with no headers, then transpose and split
    data = pd.read_csv(config["data_path"], header=None).T
    predictor_variable = data.iloc[:, 0]
    X = data.iloc[:, 1:]
    y = predictor_variable
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # pd.DataFrame.to_csv(X_test.iloc[:, 1:], './artifacts/X_test.csv')
    # Initialize feature selector and perform feature selection
    feature_selector = FeatureSelector(n_estimators=1000, random_state=random_state)
    feature_selector.fit(X_train, y_train)
    X_train_selected = feature_selector.transform(X_train)
    X_test_selected = feature_selector.transform(X_test)
    
    # Save the feature selector object
    feature_selector_path = "./artifacts/feature_selector.pkl"
    with open(feature_selector_path, "wb") as f:
        pickle.dump(feature_selector, f)
    
    # Plot top N features
    # feature_selector.plot_feature_importance(feature_names=X.columns, top_n=10)

    # Initialize model and hyperparameters based on the model name
    model_name = config["model_name"]
    if model_name == "SVC":
        model = SVC(probability=True)
        param_grid = {
            'C': config["model_params"].get("C", [0.1, 1, 5, 10, 30]),
            'kernel': config["model_params"].get("kernel", ['linear'])
        }
    elif model_name == "RandomForest":
        model = RandomForestClassifier()
        param_grid = {
            'n_estimators': config["model_params"].get("n_estimators", [100, 500, 1000, 5000]),
            'max_depth': config["model_params"].get("max_depth", [None, 10, 20]),
            'min_samples_split': config["model_params"].get("min_samples_split", [2, 5, 10])
        }
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    return X_train_selected, X_test_selected, y_train, y_test, model, param_grid


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
