import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
import seaborn as sns
from keras.callbacks import History


class ModelEvaluator:
    """
    Evaluates binary classification models using predicted probabilities.

    Input
    -----
    y_true : 1D np.ndarray of {0,1}
    y_proba: 1D np.ndarray of probabilities in [0,1]
    history: keras.callbacks.History (optional)

    Provides
    --------
    - Confusion matrix at arbitrary thresholds
    - ROC curve and AUC
    - Precision–Recall curve (+ baseline)
    - Threshold sweep (accuracy/precision/recall/F1)
    - Training curves (if history provided)
    """

    def __init__(self, y_true: np.ndarray, y_proba: np.ndarray):
        self.y_true = np.asarray(y_true).astype(int).ravel()
        self.y_proba = np.asarray(y_proba).astype(float).ravel()

        self._validate_inputs()

    def _validate_inputs(self) -> None:
        if self.y_true.ndim != 1:
            raise ValueError("y_true must be 1D")
        if self.y_proba.ndim != 1:
            raise ValueError("y_proba must be 1D")
        if len(self.y_true) != len(self.y_proba):
            raise ValueError("y_true and y_proba must have the same length")
        if not np.all((self.y_proba >= 0) & (self.y_proba <= 1)):
            raise ValueError("y_proba must be in [0, 1]")
        if not set(np.unique(self.y_true)).issubset({0, 1}):
            raise ValueError("y_true must contain only 0/1")

    def predict(self, threshold: float = 0.5) -> np.ndarray:
        return (self.y_proba >= threshold).astype(int)

    def plot_model_accuracy(self, history: History) -> None:

        hist = history.history

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        if "accuracy" in hist:
            plt.plot(hist["accuracy"], label="train accuracy", color="green")
        if "val_accuracy" in hist:
            plt.plot(hist["val_accuracy"], label="val accuracy", color="#ff4d4d")
        plt.title("Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)

        # Loss
        plt.subplot(1, 2, 2)
        plt.plot(hist["loss"], label="train loss")
        if "val_loss" in hist:
            plt.plot(hist["val_loss"], label="val loss")
        plt.title("Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()


    def plot_confusion_matrix(self, threshold: float = 0.5, normalize: bool = False) -> None:
        y_pred = self.predict(threshold)
        cm = confusion_matrix(self.y_true, y_pred)

        if normalize:
            cm_display = cm / cm.sum()
        else:
            cm_display = cm

        group_names = ["True Neg.", "False Pos.", "False Neg.", "True Pos."]
        group_counts = [f"{v:0.0f}" for v in cm.flatten()]
        group_percentages = [f"{v:.2%}" for v in (cm.flatten() / cm.sum())]

        labels = [f"{n}\n{c}\n{p}" for n, c, p in zip(group_names, group_counts, group_percentages)]
        labels = np.asarray(labels).reshape(2, 2)

        plt.figure(figsize=(4, 4))
        sns.heatmap(
            cm_display,
            annot=labels,
            fmt="",
            cmap="Blues",
            cbar=True,
            xticklabels=["Pred 0", "Pred 1"],
            yticklabels=["True 0", "True 1"],
        )
        plt.title(f"Confusion Matrix (threshold={threshold:.2f})")
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        plt.show()

    def roc_auc(self) -> float:
        fpr, tpr, _ = roc_curve(self.y_true, self.y_proba)
        return auc(fpr, tpr)

    def plot_roc_curve(self) -> None:
        fpr, tpr, _ = roc_curve(self.y_true, self.y_proba)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(5, 5))
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_precision_recall_curve(self) -> None:
        precision, recall, _ = precision_recall_curve(self.y_true, self.y_proba)
        baseline = self.y_true.mean()

        plt.figure(figsize=(5, 5))
        plt.plot(recall, precision, label="Model PR curve")
        plt.hlines(
            y=baseline,
            xmin=0,
            xmax=1,
            linestyles="--",
            colors="gray",
            label=f"Baseline (pos rate = {baseline:.2f})",
        )
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision–Recall Curve")
        plt.legend()
        plt.grid(True)
        plt.show()

    def threshold_sweep(self, thresholds: np.ndarray | None = None) -> pd.DataFrame:
        if thresholds is None:
            thresholds = np.linspace(0.01, 0.99, 99)

        rows = []
        for t in thresholds:
            y_pred = self.predict(float(t))
            rows.append({
                "threshold": float(t),
                "accuracy": accuracy_score(self.y_true, y_pred),
                "precision": precision_score(self.y_true, y_pred, zero_division=0),
                "recall": recall_score(self.y_true, y_pred, zero_division=0),
                "f1": f1_score(self.y_true, y_pred, zero_division=0),
            })

        return pd.DataFrame(rows)

    def plot_threshold_sweep(self) -> None:
        sweep = self.threshold_sweep()

        plt.figure(figsize=(8, 4))
        plt.plot(sweep["threshold"], sweep["accuracy"], label="Accuracy")
        plt.plot(sweep["threshold"], sweep["precision"], label="Precision")
        plt.plot(sweep["threshold"], sweep["recall"], label="Recall")
        plt.plot(sweep["threshold"], sweep["f1"], label="F1")
        plt.xlabel("Threshold")
        plt.ylabel("Score")
        plt.title("Threshold Sweep")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_probability_distribution(
            self,
            bins: int = 50,
            log_y: bool = False,
            show_thresholds: list[float] | None = None,
    ) -> None:
        """
        Plot side-by-side distributions of predicted probabilities
        for y_true = 0 and y_true = 1.

        Parameters
        ----------
        bins : int
            Number of histogram bins.
        log_y : bool
            If True, uses log scale on y-axis (useful for imbalance).
        show_thresholds : list[float] | None
            Optional list of thresholds to draw as vertical lines
            (e.g. [0.5, 0.48]).
        """

        proba_neg = self.y_proba[self.y_true == 0]
        proba_pos = self.y_proba[self.y_true == 1]

        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)

        # --- y = 0 ---
        axes[0].hist(
            proba_neg,
            bins=bins,
            density=True,
            alpha=0.8,
            color="red",
        )
        axes[0].set_title("Predicted probabilities | y = 0")
        axes[0].set_xlabel("Predicted probability (p̂)")
        axes[0].set_ylabel("Density")
        axes[0].grid(True)

        # --- y = 1 ---
        axes[1].hist(
            proba_pos,
            bins=bins,
            density=True,
            alpha=0.8,
            color="green",
        )
        axes[1].set_title("Predicted probabilities | y = 1")
        axes[1].set_xlabel("Predicted probability (p̂)")
        axes[1].grid(True)

        if show_thresholds:
            for t in show_thresholds:
                for ax in axes:
                    ax.axvline(
                        float(t),
                        linestyle="--",
                        linewidth=1,
                        color="black",
                        alpha=0.7,
                    )

        if log_y:
            for ax in axes:
                ax.set_yscale("log")

        for ax in axes:
            ax.set_xlim(0, 1)

        plt.tight_layout()
        plt.show()

    def summary_probability_stats(self) -> pd.DataFrame | None:
        """
        Print and optionally return summary statistics of predicted probabilities.

        Reports p10 / p50 / p90:
        - Overall
        - For true negatives (y_true == 0)
        - For true positives (y_true == 1)
        """


        rows = []

        q_all = self._quantiles(self.y_proba)
        rows.append({"group": "overall", **q_all})

        # Per class
        for cls in (0, 1):
            mask = self.y_true == cls
            if mask.any():
                q_cls = self._quantiles(self.y_proba[mask])
                rows.append({"group": f"class_{cls}", **q_cls})

        df = pd.DataFrame(rows)

        print("\nPredicted probability quantiles:")
        print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

        return df


    def compute_validation_metrics(self, threshold: float = 0.5) -> dict:
        y_pred = self.predict(threshold)
        return {
            "threshold": threshold,
            "accuracy": accuracy_score(self.y_true, y_pred),
            "precision": precision_score(self.y_true, y_pred, zero_division=0),
            "recall": recall_score(self.y_true, y_pred, zero_division=0),
            "f1": f1_score(self.y_true, y_pred, zero_division=0),
            "roc_auc": self.roc_auc(),
        }

    def _quantiles(self, x: np.ndarray) -> dict:
        return {
            "p10": np.quantile(x, 0.10),
            "p50": np.quantile(x, 0.50),
            "p90": np.quantile(x, 0.90),
        }
