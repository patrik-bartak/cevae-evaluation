import numpy as np
from matplotlib import pyplot as plt


class Grapher:

    def __init__(self, save_dir, show_graphs, save_graphs):
        self.save_dir = save_dir
        self.show_graphs = show_graphs
        self.save_graphs = save_graphs

    def scatter_2d(self, filename, x, y, x_label, y_label):
        fig, ax = plt.subplots()
        feature_one = x
        truth = y
        ax.scatter(feature_one, truth, label="Instance")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.legend()
        self.save_or_show(fig, filename)

    def scatter_2d_color(self, filename, x, y, c, x_label, y_label, c_label):
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
        ax.legend()
        cbar = fig.colorbar(scatter)
        cbar.set_label(c_label, rotation=270, labelpad=14)
        self.save_or_show(fig, filename)

    def save_or_show(self, fig, filename):
        if self.save_graphs:
            fig.savefig(self.save_dir + filename)
        if self.show_graphs:
            fig.show()
        plt.close(fig)
