import os
import pandas as pd
import numpy as np
import time
import argparse
import json

# Disable TesnorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from experiment import Experiment, Session, Parameterizer


def main():
    t = time.time_ns()
    parser = argparse.ArgumentParser(description='Enter evaluation parameters.')
    parser.add_argument('distribution', type=str)
    parser.add_argument('model_dist', type=str)
    parser.add_argument('proxy_noise', type=float)
    parser.add_argument('latent_dims', type=int)
    parser.add_argument('data_latent_dims', type=int)
    parser.add_argument('results_batch', type=str)
    parser.add_argument('description', type=str)
    parser.add_argument('data_seed', type=int)
    parser.add_argument('num_samples', type=int)
    parser.add_argument('ihdp_replication', type=int)
    parser.add_argument('ihdp_latent_feature', type=int)
    args = parser.parse_args()
    results_batch = args.results_batch
    print('STARTING...')

    print(f"Using synthetic")
    result = cevae_latent(args)

    print(type(result))
    print(result)
    save_results(result, args, results_batch)
    print(f'FINISHED IN {(time.time_ns() - t) * 1e-9} SECONDS.')


def basic_session():
    dimensions = 5
    get_experiment = lambda: Experiment() \
        .add_causal_forest(honest=False, min_leaf_size=1, number_of_trees=500) \
        .add_causal_forest(min_leaf_size=1, number_of_trees=500) \
        .add_mean_squared_error() \
        .add_biased_generator(dimensions)
    Session(get_experiment, 'basic_session').run_specific(
        pd.DataFrame(np.zeros((40, 5)), columns=[f'feature_{i}' for i in range(dimensions)]),
        pd.DataFrame(np.zeros((40, 1)) + 0.1, columns=['outcome']))


def cevae_basic_experiment():
    dimensions = 5
    sample_size = 1000
    Experiment() \
        .add_cevae(dimensions, latent_dim=1, outcome_dist="normal") \
        .add_mean_squared_error() \
        .add_spiked_generator(dimensions, sample_size) \
        .run(save_data=True, save_graphs=True, show_graphs=False)


def cevae_toy_dataset_experiment():
    data_dimensions = 1
    model_dimensions = 1
    sample_sizes = [
        {"sample_size": 10000},
        # {"sample_size": 30000},
        # {"sample_size": 50000},
    ]
    param_function = lambda d: lambda: Experiment() \
        .add_causal_forest(min_leaf_size=1, number_of_trees=500) \
        .add_cevae(model_dimensions,
                   latent_dim=5,
                   hidden_dim=50,
                   num_layers=3,
                   num_samples=100,
                   outcome_dist="bernoulli") \
        .add_all_metrics() \
        .add_easy_generator(data_dimensions, d["sample_size"]) \
        .run(save_data=True, save_graphs=True, show_graphs=False)
    Parameterizer(param_function, sample_sizes, name='cevae_toy_sum_samples').run(replications=1)


def cevae_latent(args):
    num_samples = args.num_samples
    data_dimensions = args.data_latent_dims
    # model_dimensions = 3
    model_dimensions = 3 if data_dimensions == 0 else 9
    # model_dimensions = 27
    if args.distribution == "normal":
        res = Experiment(name=args.results_batch) \
            .add_cevae(model_dimensions,
                       latent_dim=args.latent_dims,
                       hidden_dim=200,
                       num_layers=3,
                       num_samples=100,
                       batch_size=1000,
                       num_epochs=100,
                       learning_rate=1e-2,
                       learning_rate_decay=0.01,
                       weight_decay=1e-4,
                       outcome_dist="normal",
                       model_dist=args.model_dist) \
            .add_all_metrics() \
            .add_complex_latent_normal_generator(data_dimensions, num_samples, args) \
            .run(save_data=True, save_graphs=True, show_graphs=False)
    else:
        raise ValueError("Invalid latent distribution")
    return res


def cevae_easy_synthetic_experiment():
    data_dimensions = 1
    model_dimensions = 3
    sample_sizes = [
        {"proxy_noise": 0.5},
        {"proxy_noise": 0.5},
        # {"proxy_noise": 0.1},
        # {"proxy_noise": 0.5},
        # {"proxy_noise": 0.1},
        # {"proxy_noise": 0.01},
        # {"proxy_noise": 0.001},
    ]
    param_function = lambda d: lambda: Experiment() \
        .add_cevae(model_dimensions,
                   latent_dim=1,
                   hidden_dim=200,
                   num_layers=3,
                   num_samples=100,
                   batch_size=1000,
                   num_epochs=100,
                   learning_rate=1e-3,
                   learning_rate_decay=0.1,
                   weight_decay=1e-4,
                   outcome_dist="normal") \
        .add_causal_forest(min_leaf_size=1, number_of_trees=500) \
        .add_all_metrics() \
        .add_easy_generator(data_dimensions, 5000, d['proxy_noise'])
    results = Parameterizer(param_function, sample_sizes, name='cevae_baseline_noise') \
        .run(replications=1, save_data=True, save_graphs=True, show_graphs=True)
    print_results(results)


def cevae_easy_synthetic_experiment_2():
    data_dimensions = 1
    model_dimensions = 1
    sample_size = 1000
    proxy_noise = 0.2
    Experiment() \
        .add_cevae(model_dimensions,
                   latent_dim=1,
                   hidden_dim=100,
                   num_layers=3,
                   num_samples=100,
                   batch_size=1000,
                   outcome_dist="normal") \
        .add_cevae(model_dimensions,
                   latent_dim=1,
                   hidden_dim=100,
                   num_layers=3,
                   num_samples=100,
                   batch_size=1000,
                   outcome_dist="normal") \
        .add_all_metrics() \
        .add_easy_generator(data_dimensions, sample_size, proxy_noise) \
        .run(save_graphs=False, show_graphs=True)


def cevae_proxy_experiment():
    data_dimensions = 5
    model_dimensions = 9
    sample_size = 1000
    Experiment() \
        .add_cevae(model_dimensions, latent_dim=10, outcome_dist="normal") \
        .add_all_metrics() \
        .add_noisy_spiked_proxy_generator(data_dimensions, sample_size) \
        .run(save_data=True, save_graphs=True, show_graphs=False)


def basic_experiment():
    dimensions = 5
    sample_size = 50
    Experiment() \
        .add_causal_forest(honest=False, min_leaf_size=1, number_of_trees=500) \
        .add_causal_forest(min_leaf_size=1, number_of_trees=500) \
        .add_mean_squared_error() \
        .add_biased_generator(dimensions=dimensions, sample_size=sample_size) \
        .run(save_data=True, save_graphs=True, show_graphs=True)


def print_results(results):
    for param, table in results:
        print(table)


def save_results(result, args, results_batch):
    shortened_arg_vals = shorten_args(args.__dict__)
    arg_filename = "_".join(shortened_arg_vals)
    result_dir = f"results/{results_batch}/{arg_filename}"
    os.makedirs(result_dir, exist_ok=True)
    desc_path = f"{result_dir}/config.json"
    with open(desc_path, "w") as f:
        f.write(json.dumps(args.__dict__, indent=2))
        f.write("\n")
    result_t = result.head(1)
    output_path = f"{result_dir}/cevae_{arg_filename}.csv"
    result_t.to_csv(output_path, mode='a', header=not os.path.exists(output_path))


def shorten_args(argument_names):
    f_names = []
    for a_n, val in argument_names.items():
        if a_n in ["data_seed", "ihdp_replication", "ihdp_latent_feature"]:
            continue
        split = a_n.split("_")
        shortened = []
        for c in split:
            shortened.append(c[0])
        f_names.append(f"{''.join(shortened)}={val}")
    return f_names


if __name__ == '__main__':
    main()
