from sklearn.model_selection import GridSearchCV
import joblib

class Trainer:
    def __init__(self, model, param_grid, cv=5, scoring="accuracy"):
        """
        Initialize the Trainer class with a model and its parameter grid.
        """
        self.model = model
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring
        self.best_model = None

    def train(self, X_train, y_train):
        """
        Train the model using GridSearchCV for hyperparameter tuning.
        """
        grid_search = GridSearchCV(estimator=self.model, param_grid=self.param_grid, cv=self.cv, scoring=self.scoring)
        grid_search.fit(X_train, y_train)
        self.best_model = grid_search.best_estimator_
        print(f"Best parameters found: {grid_search.best_params_}")
        return self.best_model

    def save_model(self, filepath):
        """
        Save the trained model to a file.
        """
        if self.best_model:
            joblib.dump(self.best_model, filepath)
        else:
            print("No model to save. Please train the model first.")
