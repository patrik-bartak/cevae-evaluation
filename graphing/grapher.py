import numpy as np
from matplotlib import pyplot as plt


class Grapher:
    """
    Graphing object that generates plots from data
    """

    def __init__(self, save_dir, show_graphs, save_graphs):
        """
        Inits new object
        :param save_dir: Directory to save plots
        :param show_graphs: Whether to show plots
        :param save_graphs: Whether to save plots
        """
        self.save_dir = save_dir
        self.show_graphs = show_graphs
        self.save_graphs = save_graphs

    def pred_actual_scatter(self, filename, x, y, x_label, y_label, indicate_treated=None):
        """
        Plot actual ITE vs predicted ITE in a scatter plot
        :param filename: Name of plot
        :param x: X axis
        :param y: Y axis
        :param x_label: X label
        :param y_label: Y lable
        :param indicate_treated: Whether treated should be a distinct colour
        :return:
        """
        fig, ax = plt.subplots()
        if indicate_treated is not None:
            t_1s_idxs, = np.nonzero(indicate_treated.array)
            t_0s_idxs, = np.nonzero(1 - indicate_treated.array)
            actual_t_1 = x[t_1s_idxs]
            actual_t_0 = x[t_0s_idxs]
            preds_t_1 = y[t_1s_idxs]
            preds_t_0 = y[t_0s_idxs]
            ax.scatter(actual_t_1, preds_t_1, label="Data instance (t=1)", color="aquamarine", alpha=0.5, s=4**2)
            ax.scatter(actual_t_0, preds_t_0, label="Data instance (t=0)", color="orange", alpha=0.5, s=4**2)
            assert len(actual_t_1) == len(preds_t_1)
            assert len(actual_t_1) + len(actual_t_0) == len(x)
        else:
            actual = x
            preds = y
            ax.scatter(actual, preds, label="Data instance", color="blue", alpha=0.5, s=4 ** 2)
        ideal = np.linspace(-100, 100, 200)
        ax.axis('equal')
        ax.autoscale(False)
        ax.plot(ideal, ideal, label="Perfect model", color="0.5", alpha=0.5,  linestyle='dashed')
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(color="0.9")
        ax.legend()
        self.save_or_show(fig, filename)

    def kde(self, filename, vals):
        """
        Kernel density estimation plot of values
        :param filename: Name of plot
        :param vals: Values to plot
        :return:
        """
        from sklearn.neighbors import KernelDensity
        try:
            num_col = len(vals.columns)
        except Exception as e:
            # print(e)
            num_col = 1
        vals = vals.values
        mn = min(vals)
        mx = max(vals)
        vals = np.reshape(vals, (-1, num_col))
        fig, ax = plt.subplots()
        # instantiate and fit the KDE model
        kde = KernelDensity(bandwidth=0.01, kernel='tophat')
        kde.fit(vals)

        test = np.linspace(mn, mx, 1000)[:, np.newaxis]
        # score_samples returns the log of the probability density
        logprob = kde.score_samples(test)

        ax.fill_between(np.reshape(test, -1), np.exp(logprob), alpha=0.5)
        ax.plot(vals, np.full_like(vals, -0.01), '|k', markeredgewidth=1)
        ax.grid(color="0.9")
        # ax.set_ylim(0, 1)
        self.save_or_show(fig, filename)

    def plot_losses(self, filename, losses_dict):
        for loss_name, loss_values in losses_dict.items():
            fig, ax = plt.subplots()
            if type(loss_values) is dict:
                for line_name, actual_values in loss_values.items():
                    epochs = np.arange(len(actual_values))
                    ax.plot(epochs, actual_values, label=line_name)
            else:
                epochs = np.arange(len(loss_values))
                ax.plot(epochs, loss_values)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(loss_name)
            ax.grid(color="0.9")
            ax.legend()
            file_loss_name = loss_name.lower().replace(" ", "_")
            self.save_or_show(fig, f"{filename}_{file_loss_name}")

    def scatter_2d(self, filename, x, y, x_label, y_label):
        """
        2d Scatter plot
        :param filename: Name of plot
        :param x: X axis
        :param y: Y axis
        :param x_label: X label
        :param y_label: Y label
        :return:
        """
        fig, ax = plt.subplots()
        feature_one = x
        truth = y
        ax.scatter(feature_one, truth, label="Instance")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        # ax.set_xlim([0.0, 1.0])
        # ax.set_ylim([0.0, 1.0])
        ax.grid(color="0.9")
        ax.legend()
        self.save_or_show(fig, filename)

    def line_2d(self, filename, x, y, x_label, y_label):
        """
        2d Line plot
        :param filename: Name of plot
        :param x: X axis
        :param y: Y axis
        :param x_label: X label
        :param y_label: Y label
        :return:
        """
        fig, ax = plt.subplots()
        feature_one = x
        truth = y
        ax.plot(feature_one, truth)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(color="0.9")
        ax.legend()
        self.save_or_show(fig, filename)

    def scatter_2d_color(self, filename, x, y, c, x_label, y_label, c_label):
        """
        2d Scatter coloured plot
        :param filename: Name of plot
        :param x: X axis
        :param y: Y axis
        :param c: C values
        :param x_label: X label
        :param y_label: Y label
        :param c_label: C label
        :return:
        """
        fig, ax = plt.subplots()
        feature_one = x
        feature_two = y
        maximal = c.max()
        minimal = c.min()
        color_function = lambda i: [0,
                                    np.clip((c.iloc[i] - minimal) / (maximal - minimal + 0.01), 0, 1),
                                    np.clip(1 - (c.iloc[i] - minimal) / (maximal - minimal + 0.01), 0, 1)]
        scatter = ax.scatter(feature_one, feature_two, c=[color_function(i) for i in c.index], label="Instance")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(color="0.9")
        ax.legend()
        cbar = fig.colorbar(scatter)
        cbar.set_label(c_label, rotation=270, labelpad=14)
        self.save_or_show(fig, filename)

    def save_or_show(self, fig, filename):
        """
        Helper function to save of show plots
        :param fig: Figure object to save or show
        :param filename: Name of file to save or show
        :return:
        """
        if self.save_graphs:
            fig.savefig(self.save_dir + filename)
        if self.show_graphs:
            fig.show()
        plt.close(fig)
