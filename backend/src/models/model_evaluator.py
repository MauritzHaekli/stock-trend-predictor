import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report


class ModelEvaluator:
    def __init__(self, test_loss, y_test, y_pred):
        self.test_loss = test_loss
        self.accuracy: float = accuracy_score(y_test, y_pred)
        self.precision: float = precision_score(y_test, y_pred)
        self.recall: float = recall_score(y_test, y_pred)
        self.f1: float = f1_score(y_test, y_pred)
        try:
            self.roc_auc: float = roc_auc_score(y_test, y_pred)
        except ValueError as error:
            if "Only one class present in y_true" in str(error):
                self.roc_auc: float = 0
                print("ROC AUC score is not defined when only one class is present.")
            else:
                raise error

        self.evaluation_metrics: pd.DataFrame = self.get_evaluation_metrics()

    def get_evaluation_metrics(self) -> pd.DataFrame:
        metrics_data = {
            'Metric': ['Loss', 'Accuracy', 'Precision', 'Recall', 'F1Score', 'ROC AUC'],
            'Score': [self.test_loss, self.accuracy, self.precision, self.recall, self.f1, self.roc_auc]
        }

        return pd.DataFrame(metrics_data)

