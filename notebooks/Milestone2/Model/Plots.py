from sklearn.calibration import CalibrationDisplay
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter
from sklearn.metrics import roc_curve, auc
import numpy as np

from notebooks.Milestone2.common.save_plot import save_plot


class Plots:
    def __init__(self, models: list) -> None:
        self.models = models

    def save_and_show_plots(self, base_file_name: str, directory: str):
        self.make_roc_plot()
        save_plot(f'{base_file_name} ROC', directory)
        self.make_goal_rate_plot()
        save_plot(f'{base_file_name} Goal Rates', directory)
        self.make_cumul_perc_goal_plot()
        save_plot(f'{base_file_name} Cumulative Percentage Of Goals', directory)
        self.make_reliability_plot()
        save_plot(f'{base_file_name} Reliability', directory)
        plt.show()

    def show_plots(self):
        self.make_roc_plot()
        self.make_goal_rate_plot()
        self.make_cumul_perc_goal_plot()
        self.make_reliability_plot()
        plt.show()

    def make_roc_plot(self):  # to plot Q3 figure a
        plt.figure()
        lw = 2

        for model in self.models:
            fpr = {}
            tpr = {}
            curve_auc = {}
            Y_hat = model.goal_probability()
            Y_val = model.Y_val
            fpr["pred"], tpr["pred"], _ = roc_curve(Y_val, Y_hat, pos_label=1)
            curve_auc["pred"] = auc(fpr["pred"], tpr["pred"])
            plt.plot(fpr["pred"], tpr["pred"], lw=lw,
                     label=f"ROC curve (area = %0.2f) for {model.Name}" % curve_auc["pred"])

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver operating characteristic AUC")
        plt.legend()
        return plt

    def make_goal_rate_plot(self):  # to plot Q3 figure b
        plt.figure()

        def goal_rate(y_hat, y_val, perc=None):
            if perc is None:
                out = y_hat[y_hat == 1].shape[0] / y_hat.shape[0]
            else:
                n_tot = y_hat[y_hat >= [perc]].shape[0]
                n_goals = np.sum(y_hat[y_val == 1] >= perc)
                out = n_goals / n_tot
            return out

        percentiles = np.linspace(100, 0, 11)

        for model in self.models:
            out = []
            Y_hat = model.goal_probability()
            for p in percentiles:
                perc = np.percentile(Y_hat, p)
                n_goal_perc = goal_rate(Y_hat, model.Y_val, perc)
                out.append(n_goal_perc)

            plt.plot(percentiles, out, lw=2, label=f"{model.Name}")
        plt.xlim([101, -1])
        plt.ylim([-0.05, 1.05])
        plt.xticks(percentiles)
        plt.yticks(np.linspace(0, 1, 11))
        plt.xlabel("Shot Probability Model Percentile")
        plt.ylabel("Goal Rate %")
        plt.title("Goal Rate")
        plt.legend()
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
        plt.gca().xaxis.set_major_formatter(PercentFormatter(100))
        return plt

    def make_cumul_perc_goal_plot(self):  # to plot Q3 figure c
        plt.figure()  # Plotting 3c

        for model in self.models:
            x, y = self.__split_goal_percentiles(model)
            plt.plot(x, y, lw=2, label=f"{model.Name}")

        plt.xlim([101, -1])
        plt.ylim([-0.05, 1.05])
        plt.xticks(x)
        plt.yticks(np.linspace(0, 1, 11))
        plt.xlabel("Shot Probability Model Percentile")
        plt.ylabel("Proportion")
        plt.title("Cumulative % of goals")
        plt.legend(loc="lower right")
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
        plt.gca().xaxis.set_major_formatter(PercentFormatter(100))

        return plt

    def make_reliability_plot(self):
        disp = None
        for model in self.models:
            Y_hat = model.goal_probability()
            Y_val = model.Y_val
            if disp:
                disp = CalibrationDisplay.from_predictions(Y_val, Y_hat, strategy='quantile', ax=disp.ax_,
                                                           name=model.Name)
            else:
                disp = CalibrationDisplay.from_predictions(Y_val, Y_hat, strategy='quantile', name=model.Name)

        plt.legend(loc="upper left")
        plt.title("Baseline Models' Calibration Curves")
        return plt

    def __split_goal_percentiles(self, model):
        Y_val = model.Y_val
        Y_hat = model.goal_probability()
        Y_hat_goals = Y_hat[Y_val == 1]
        n_goals = Y_hat_goals.shape[0]
        percentiles = np.linspace(100, 0, 11)
        cum_prop = []
        for p in percentiles:
            perc = np.percentile(Y_hat, p)
            n_goal_perc = np.sum(Y_hat_goals >= perc)
            cum_prop.append(n_goal_perc / n_goals)
        return percentiles, cum_prop
