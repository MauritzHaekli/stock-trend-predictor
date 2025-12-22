import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
import pandas as pd


def plot_model_accuracy(accuracy_history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(accuracy_history.history['accuracy'][0:], label='Training Accuracy', color="green")
    ax1.plot(accuracy_history.history['val_accuracy'][0:], label='Validation Accuracy', color="#ff4d4d")

    ax1.set_title('Accuracy Plot')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()

    ax2.plot(accuracy_history.history['loss'][1:], label='Training Loss', color="green")
    ax2.plot(accuracy_history.history['val_loss'][1:], label='Validation Loss', color="#ff4d4d")

    ax2.set_title('Loss Plot')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(target_test_data, target_prediction_data):
    conf_matrix = confusion_matrix(target_test_data, target_prediction_data)

    plt.figure(figsize=(4, 4))

    group_names = ["True Neg.", "False Pos.", "False Neg.", "True Pos."]
    group_counts = ["{0:0.0f}".format(value) for value in conf_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in conf_matrix.flatten() / np.sum(conf_matrix)]

    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)

    sns.heatmap(conf_matrix, annot=labels, fmt='', cmap='Blues', cbar=True, xticklabels=['Predicted 0', 'Predicted 1'],
                yticklabels=['Actual 0', 'Actual 1'])
    plt.title('Confusion Matrix')
    plt.show()


def plot_correlation_heatmap(dataframe: pd.DataFrame):
    plt.figure(figsize=(32, 16))
    sns.heatmap(dataframe.corr(), annot=True, cmap="YlGnBu", fmt=".2f")
