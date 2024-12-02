import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt

class FeatureSelector:
    def __init__(self, n_estimators=1000, random_state=42, n_jobs=-1):
        # Initialize the random forest classifier with given hyperparameters
        self.rf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=n_jobs)
        self.selector = None
        self.importances = None

    def fit(self, X_train, y_train, threshold='mean'):
        """
        Train the RandomForest model for feature importance and apply feature selection.
        
        Parameters:
        - X_train: Training features
        - y_train: Training labels
        - threshold: Threshold for feature selection (default is 'mean')
        
        Returns:
        - self
        """
        # Fit the model and calculate feature importances
        self.rf.fit(X_train, y_train)
        self.importances = self.rf.feature_importances_

        # Select features based on importance threshold
        self.selector = SelectFromModel(self.rf, threshold=threshold, prefit=True)
        return self

    def transform(self, X):
        """
        Transform the dataset to keep only selected features.
        
        Parameters:
        - X: Features to transform
        
        Returns:
        - Transformed feature set
        """
        if not self.selector:
            raise RuntimeError("FeatureSelector must be fit before calling transform.")
        return self.selector.transform(X)

    def plot_feature_importance(self, feature_names, top_n=10):
        """
        Plot the top N feature importances.
        
        Parameters:
        - feature_names: List of feature names
        - top_n: Number of top features to display
        """
        if self.importances is None:
            raise RuntimeError("Feature importances not available. Call fit() first.")
        
        # Sort features by importance
        forest_importances = pd.Series(self.importances, index=feature_names)
        forest_importances_sorted = forest_importances.sort_values(ascending=False)

        # Get top N features and plot them
        top_features = forest_importances_sorted[:top_n]
        std = np.std([tree.feature_importances_ for tree in self.rf.estimators_], axis=0)
        top_std = std[forest_importances_sorted.index[:top_n]]

        fig, ax = plt.subplots()
        top_features.plot.bar(yerr=top_std, ax=ax)
        ax.set_title(f"Top {top_n} Feature Importances using MDI")
        ax.set_ylabel("Mean decrease in impurity")
        fig.tight_layout()
        plt.show()
