import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report


class ModelEvaluator:
    def __init__(self, test_loss, y_test, y_pred):
        self.test_loss = test_loss
        self.accuracy = accuracy_score(y_test, y_pred)
        self.precision = precision_score(y_test, y_pred)
        self.recall = recall_score(y_test, y_pred)
        self.f1 = f1_score(y_test, y_pred)
        self.roc_auc = roc_auc_score(y_test, y_pred)

        self.evaluation_metrics = self.get_metrics_data()

    def get_metrics_data(self):
        metrics_data = {
            'Metric': ['Loss', 'Accuracy', 'Precision', 'Recall', 'F1Score', 'ROC AUC'],
            'Score': [self.test_loss, self.accuracy, self.precision, self.recall, self.f1, self.roc_auc]
        }

        return pd.DataFrame(metrics_data)

