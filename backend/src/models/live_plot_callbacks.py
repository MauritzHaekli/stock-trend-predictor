import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from keras.callbacks import Callback
from IPython.display import display, update_display


class LivePlotCallback(Callback):
    """
    PyCharm-safe live training plot callback.
    LivePlotCall modifies the Callback class of keras.callbacks.
    The methods on_epoch_begin and on_epoch_end cannot be renamed.

    - Displays ONE continuously updating plot
    - Minimizes flicker by rendering at epoch begin
    """

    def __init__(self, update_every: int = 1):
        super().__init__()
        self.update_every = update_every

        self.accuracy = []
        self.val_accuracy = []

        self.loss = []
        self.val_loss = []

        self.precision = []
        self.val_precision = []

        self.recall = []
        self.val_recall = []

        self.auc = []
        self.val_auc = []

        self.pr_auc = []
        self.val_pr_auc = []

        self.display_id: str = "training_progress"
        self._initialized: bool = False

        self.train_color: str = "red"
        self.val_color: str = "green"

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        self.accuracy.append(logs.get("accuracy"))
        self.val_accuracy.append(logs.get("val_accuracy"))

        self.loss.append(logs.get("loss"))
        self.val_loss.append(logs.get("val_loss"))

        self.precision.append(logs.get("precision"))
        self.val_precision.append(logs.get("val_precision"))

        self.recall.append(logs.get("recall"))
        self.val_recall.append(logs.get("val_recall"))

        self.auc.append(logs.get("auc"))
        self.val_auc.append(logs.get("val_auc"))

        self.pr_auc.append(logs.get("pr_auc"))
        self.val_pr_auc.append(logs.get("val_pr_auc"))

    def on_epoch_begin(self, epoch, logs=None):
        if epoch == 0 or epoch % self.update_every != 0:
            return

        fig = plt.figure(figsize=(13, 9), dpi=110)
        gs = gridspec.GridSpec(2, 3, hspace=0.15, wspace=0.2)

        axes = [fig.add_subplot(gs[0, 0]),
                fig.add_subplot(gs[0, 1]),
                fig.add_subplot(gs[0, 2]),
                fig.add_subplot(gs[1, 0]),
                fig.add_subplot(gs[1, 1]),
                fig.add_subplot(gs[1, 2]),
                ]

        fig.suptitle(
            f"Training Progress (Epoch {epoch})",
            fontsize=14,
            weight="bold",
        )

        fig.subplots_adjust(
            top=0.92,
            bottom=0.07,
            left=0.07,
            right=0.98,
        )

        self._plot_metric(axes[0], self.accuracy, self.val_accuracy,"Accuracy")
        self._plot_metric(axes[3], self.loss, self.val_loss,"Loss")
        self._plot_metric(axes[1], self.precision, self.val_precision,"Precision")
        self._plot_metric(axes[4], self.recall, self.val_recall,"Recall")
        self._plot_metric(axes[2], self.auc, self.val_auc,"ROC AUC")
        self._plot_metric(axes[5], self.pr_auc, self.val_pr_auc,"PR-AUC")

        fig.suptitle(
            f"Training Progress (Epoch {epoch})",
            fontsize=14,
            weight="bold",
        )

        if not self._initialized:
            display(fig, display_id=self.display_id)
            self._initialized = True
        else:
            update_display(fig, display_id=self.display_id)

        plt.close(fig)

    def _plot_metric(self,ax,train,val,title: str,ylim: tuple[float, float] | None = None):
        ax.plot(train, label="train", linewidth=2, color=self.train_color)
        ax.plot(val, label="val", linewidth=2, color=self.val_color)
        ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", frameon=False, fontsize=9)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        if ylim is not None:
            ax.set_ylim(*ylim)







