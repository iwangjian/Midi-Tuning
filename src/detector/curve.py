import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics, calibration


def plot_roc_curve(y_trues, y_probs):
    assert len(y_trues) == len(y_probs)
    plt.figure("ROC Curve", figsize=(8, 6))
    ax = plt.subplot()
    fpr, tpr, thresholds  = metrics.roc_curve(y_trues, y_probs, pos_label=1)
    auc_score = metrics.auc(fpr, tpr)
    ax.plot(fpr, tpr, "-", label="Estimator (AUC = {:.2f})".format(auc_score))
    ax.plot([0, 1], [0, 1], "k:", label="Chance Level (AUC = 0.5))")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlabel("False Positive Rate")
    ax.legend(loc="lower right")
    ax.set_title("ROC Curve")
    x_ticks = np.arange(0.0, 1.05, 0.2)
    y_ticks = np.arange(0.0, 1.05, 0.2)
    plt.xticks(x_ticks)
    plt.yticks(y_ticks)
    plt.grid(ls='--')
    plt.show()

def plot_calibration_curve(y_trues, y_probs, n_bins=10):
    assert len(y_trues) == len(y_probs)
    plt.figure("Calibration Curve", figsize=(8, 6))
    ax = plt.subplot()

    fraction_of_positives, mean_predicted_value = calibration.calibration_curve(y_trues, y_probs, n_bins=n_bins)
    ax.plot(mean_predicted_value, fraction_of_positives, "s-g", label="Estimator")
    ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    ax.set_ylabel("Fraction of Positives")
    ax.set_xlabel("Predicted Prob. of Positives")
    ax.legend(loc="lower right")
    ax.set_title("Calibration Curve")
    x_ticks = np.arange(0.0, 1.05, 0.2)
    y_ticks = np.arange(0.0, 1.05, 0.2)
    plt.xticks(x_ticks)
    plt.yticks(y_ticks)
    plt.show()
