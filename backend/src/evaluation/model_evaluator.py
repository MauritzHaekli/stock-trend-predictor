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
from sklearn.calibration import calibration_curve
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
                "f1": f1_score(self.y_true, y_pred, zero_division=0)
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

    def plot_probability_distribution(self, bins: int = 50, log_y: bool = False, show_thresholds: list[float] | None = None) -> None:
        """
        Plot side-by-side distributions of predicted probabilities
        for y_true = 0 and y_true = 1.

        bins : int Number of histogram bins.
        log_y : bool If True, uses log scale on y-axis (useful for imbalance).
        show_thresholds : list[float] | None Optional list of thresholds to draw as vertical lines
        """

        proba_neg = self.y_proba[self.y_true == 0]
        proba_pos = self.y_proba[self.y_true == 1]

        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)

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

    def summary_probability_stats(self) -> pd.DataFrame:
        """
        Print and return quantile statistics (p10 / p50 / p90)
        for predicted probabilities overall and per class.
        """

        rows = {
            "overall": self._quantiles(self.y_proba),
            "y=0": self._quantiles(self.y_proba[self.y_true == 0]),
            "y=1": self._quantiles(self.y_proba[self.y_true == 1]),
        }

        df = pd.DataFrame(rows).T
        return df.round(4)

    def plot_probability_kde(
            self,
            show_quantiles: bool = True,
            quantiles: tuple[float, float, float] = (0.1, 0.5, 0.9),
    ):
        """
        Plot KDE distributions of predicted probabilities for each class
        in side-by-side subplots.
        """

        proba_neg = self.y_proba[self.y_true == 0]
        proba_pos = self.y_proba[self.y_true == 1]

        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

        sns.kdeplot(proba_neg, ax=axes[0], fill=True, color="red")
        axes[0].set_title(f"y=0 (n={len(proba_neg)})")
        axes[0].set_xlabel("Predicted probability")

        sns.kdeplot(proba_pos, ax=axes[1], fill=True, color="green")
        axes[1].set_title(f"y=1 (n={len(proba_pos)})")
        axes[1].set_xlabel("Predicted probability")

        if show_quantiles:
            cmap = plt.get_cmap("inferno")
            colors = cmap(np.linspace(0, 1, len(quantiles)))

            for quantile, color in zip(quantiles, colors):
                axes[0].axvline(
                    np.quantile(proba_neg, quantile),
                    linestyle="--",
                    color=color,
                    alpha=0.8,
                    label=f"q={quantile:.2f}"
                )
                axes[1].axvline(
                    np.quantile(proba_pos, quantile),
                    linestyle="--",
                    color=color,
                    alpha=0.8,
                    label=f"q={quantile:.2f}"
                )

        axes[0].legend()
        axes[1].legend()

        for ax in axes:
            ax.set_xlim(0, 1)
            ax.grid(True)

        plt.tight_layout()
        plt.show()

    def plot_calibration_curve(self, n_bins: int = 10):
        """
        Plot calibration (reliability) curve.
        """

        frac_pos, mean_pred = calibration_curve(
            self.y_true,
            self.y_proba,
            n_bins=n_bins,
            strategy="uniform",
        )

        plt.figure(figsize=(5, 5))
        plt.plot(mean_pred, frac_pos, marker="o", label="Model")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")

        plt.xlabel("Mean predicted probability")
        plt.ylabel("Observed positive rate")
        plt.title("Calibration Curve (Reliability Diagram)")
        plt.legend()
        plt.grid(True)
        plt.show()

    def expected_precision_at_threshold(self, threshold: float) -> float:
        """
        Expected precision when acting only on predictions >= threshold.
        """
        y_pred = self.predict(threshold)
        mask = y_pred == 1

        if mask.sum() == 0:
            return 0.0

        return precision_score(self.y_true[mask], y_pred[mask], zero_division=0)

    def expected_calibration_error(self, n_bins: int = 10) -> float:
        """
        Expected Calibration Error (ECE).
        """

        bins = np.linspace(0, 1, n_bins + 1)
        bin_ids = np.digitize(self.y_proba, bins) - 1

        ece = 0.0
        for i in range(n_bins):
            mask = bin_ids == i
            if mask.sum() == 0:
                continue

            acc = self.y_true[mask].mean()
            conf = self.y_proba[mask].mean()
            ece += np.abs(acc - conf) * (mask.sum() / len(self.y_true))

        return ece

    def compute_validation_metrics(self, threshold: float = 0.5) -> dict:
        y_pred = self.predict(threshold)
        return {
            "threshold": threshold,
            "accuracy": accuracy_score(self.y_true, y_pred),
            "precision": precision_score(self.y_true, y_pred, zero_division=0),
            "recall": recall_score(self.y_true, y_pred, zero_division=0),
            "f1": f1_score(self.y_true, y_pred, zero_division=0),
            "roc_auc": self.roc_auc()
        }

    def compute_precision_scores(self):
        return {
            "10%": self.expected_precision_at_threshold(threshold=0.1),
            "50%": self.expected_precision_at_threshold(threshold=0.5),
            "90%": self.expected_precision_at_threshold(threshold=0.9),
        }

    def _quantiles(self, x):
        return {
            "p10": np.quantile(x, 0.10),
            "p50": np.quantile(x, 0.50),
            "p90": np.quantile(x, 0.90),
            "mean": np.mean(x),
        }
