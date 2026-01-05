import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from keras.callbacks import Callback
from IPython.display import display, update_display


class LivePlotCallback(Callback):
    def __init__(self, update_every: int = 1):
        super().__init__()
        self.update_every = update_every

        self.metrics = {
            "Accuracy": ("accuracy", "val_accuracy"),
            "Precision": ("precision", "val_precision"),
            "ROC AUC": ("auc", "val_auc"),
            "Loss": ("loss", "val_loss"),
            "Recall": ("recall", "val_recall"),
            "PR AUC": ("pr_auc", "val_pr_auc"),
        }

        self.history = {k: ([], []) for k in self.metrics}
        self.display_id = "training_progress"

        self.train_color = "red"
        self.val_color = "green"

        self.fig = None
        self.axes = []
        self.lines = {}

    def on_train_begin(self, logs=None):
        self.fig = plt.figure(figsize=(11, 7), dpi=110)
        gs = gridspec.GridSpec(2, 3, hspace=0.15, wspace=0.2)

        titles = list(self.metrics.keys())
        self.axes = [self.fig.add_subplot(gs[i // 3, i % 3]) for i in range(6)]

        for ax, title in zip(self.axes, titles):
            train_line, = ax.plot([], [], color=self.train_color, linewidth=2, label="train")
            val_line, = ax.plot([], [], color=self.val_color, linewidth=2, label="val")

            ax.set_title(title, fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.legend(frameon=False, fontsize=9)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))

            self.lines[title] = (train_line, val_line)

        self.fig.suptitle("Training Progress", fontsize=14, weight="bold")
        self.fig.subplots_adjust(top=0.92, bottom=0.07, left=0.07, right=0.98)

        display(self.fig, display_id=self.display_id)

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.update_every != 0:
            return

        for title, (train_key, val_key) in self.metrics.items():
            train_hist, val_hist = self.history[title]
            train_hist.append(logs.get(train_key))
            val_hist.append(logs.get(val_key))

            train_line, val_line = self.lines[title]
            train_line.set_data(range(len(train_hist)), train_hist)
            val_line.set_data(range(len(val_hist)), val_hist)

        for ax in self.axes:
            ax.relim()
            ax.autoscale_view()

        self.fig.suptitle(f"Training Progress (Epoch {epoch})", fontsize=14, weight="bold")
        update_display(self.fig, display_id=self.display_id)

    def on_train_end(self, logs=None):

        update_display(self.fig, display_id=self.display_id)
        plt.close(self.fig)








