import pandas as pd
import matplotlib.pyplot as plt
import glob
import numpy as np
import os


def parse_ss_label(s):
    # \\ because it is last
    return int(s.split("=")[8].split("\\")[0])


def parse_ld_label(s):
    # _ because it is not last
    return int(s.split("=")[4].split("_")[0])


def parse_md_label(s):
    # _ because it is not last
    return s.split("=")[2].split("_")[0]


def multiplot(file_regex, save_path, data_metric, y_lims, mode="ld"):
    plot_data = []
    x_labels = []
    files = glob.glob(file_regex)
    try:
        if mode == "ss":
            files = list(sorted(files, key=parse_ss_label))
        else:
            files = list(sorted(files, key=parse_ld_label))
    except ValueError:
        files = list(sorted(files, key=parse_md_label))
    print("Regex: ", file_regex)
    for data_path in files:
        data = pd.read_csv(data_path)
        data = data[data_metric]
        print(f"{len(data)} ", end="")
        plot_data.append(data)
        try:
            if mode == "ss":
                x_labels.append(parse_ss_label(data_path))
            else:
                x_labels.append(parse_ld_label(data_path))
        except ValueError:
            x_labels.append(parse_md_label(data_path))
    print()
    fig1, ax1 = plt.subplots()
    # ax1.set_title(f"{save_path}")
    medianprops = dict(color='purple')
    ax1.boxplot(plot_data, medianprops=medianprops)
    ax1.grid(color="0.9")
    if type(x_labels[0]) is int:
        if mode == "ss":
            ax1.set_xlabel("Sample size")
        else:
            ax1.set_xlabel("Number of Model Latent Dimensions")
    else:
        ax1.set_xlabel("Model Latent Distribution")
    ax1.set_ylabel(metrics_labels[data_metric])
    ax1.set_ylim(y_lims)
    ax1.set_xticks(np.arange(1, len(plot_data) + 1), labels=x_labels)
    fig1.set_size_inches(7.4, 4.8)
    plt.savefig(f"{save_path}-{y_lims}.png", bbox_inches='tight')
    plt.show()


# multiplot("results/d=normal_pn=0.2_ld=*/cevae_d=normal_pn=0.2_ld=*.csv", "results/mse-changing-ld-1-less-proxies.jpg")
# multiplot("results/3_proxies_instead_9/d=normal_pn=0.2_ld=*/cevae_d=normal_pn=0.2_ld=*.csv", "results/3_proxies_instead_9/mse-changing-ld-1.jpg")
# multiplot("results/ihdp-normal-diff-latent/*/*.csv", "results/ihdp-normal-diff-latent/mse-changing-ld-1.jpg")
# multiplot("results/normal-diff-latent-distribution/*/*.csv", "results/normal-diff-latent-distribution/mse-changing-ld-1.jpg")
# multiplot("results/bernoulli-diff-latent-distribution/*/*.csv", "results/bernoulli-diff-latent-distribution/mse-changing-ld-1.jpg")
# multiplot("results/1-latent-diff-dim/*/*.csv", "results/1-latent-diff-dim/mse-changing-ld-1.jpg")

metrics_labels = {
    "eATE": "eATE",
    "PEHE (MSE)": "PEHE (MSE)",
    "PEHE (MAE)": "MAE",
    "PEHE (RMSE)": "Square root of PEHE (RMSE)",
}

metrics = {
    "eate": "eATE",
    "mse": "PEHE (MSE)",
    "mae": "PEHE (MAE)",
    "rmse": "PEHE (RMSE)",
}

for k, v in metrics.items():
    for experiment_dir in glob.glob("results/*/"):
        for y_lims in [[0.0, None], [None, None]]:
            # for run in glob.glob(os.path.join(experiment_dir, "*", "")):
            #     print(run)
            multiplot(
                os.path.join(experiment_dir, "*", "*.csv"),
                os.path.join(experiment_dir, f"{k}"),
                v,
                y_lims,
                mode="ss"  # ss for sample size or ld for latent dimensionality
            )
