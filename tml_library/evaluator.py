from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

class Evaluator:
    def __init__(self, model):
        """
        Initialize the Evaluator with a trained model.
        """
        self.model = model

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model and return metrics.
        """
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]  # For ROC AUC and curve

        # Accounts for datatypes of target column
        if isinstance(y_test.iloc[0], str):
            pos_label = "Tumor"
        else:
            pos_label = 1.0

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, pos_label=pos_label),
            "recall": recall_score(y_test, y_pred, pos_label=pos_label),
            "f1_score": f1_score(y_test, y_pred, pos_label=pos_label),
            "roc_auc": roc_auc_score(y_test, y_pred_proba)
        }
        
        return metrics

    def plot_roc_curve(self, X_test, y_test):
        """
        Plot the ROC curve.
        """
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        plt.figure()
        plt.plot(fpr, tpr, label="ROC Curve (AUC = {:.2f})".format(roc_auc_score(y_test, y_pred_proba)))
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="best")
        plt.show()
