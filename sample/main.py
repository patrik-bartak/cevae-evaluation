import os
import pandas as pd
import numpy as np
import time

# Disable TesnorFlow warnings
from sample.bayesian_optimizer import BayesianOptimizer

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from experiment import Experiment
from session import Session
from parameterizer import Parameterizer


def main():
    t = time.time_ns()
    print('STARTING...')
    cevae_easy_synthetic_experiment()
    print(f'FINISHED IN {(time.time_ns() - t) * 1e-9} SECONDS.')


def parameterize_sample_size_biased():
    dimensions = 5
    sample_sizes = [
        {'sample_size': 50},
        {'sample_size': 100},
        {'sample_size': 250},
        {'sample_size': 500},
        {'sample_size': 750},
        {'sample_size': 1000},
        {'sample_size': 1250},
        {'sample_size': 1500},
        {'sample_size': 1750},
        {'sample_size': 2000}
    ]
    param_function = lambda d: lambda: Experiment() \
        .add_causal_forest(honest=False, min_leaf_size=1, number_of_trees=500) \
        .add_causal_forest(min_leaf_size=1, number_of_trees=500) \
        .add_mean_squared_error() \
        .add_biased_generator(dimensions=dimensions, sample_size=d['sample_size'])
    Parameterizer(param_function, sample_sizes, name='sample_size_biased_general').run(save_graphs=True)
    Parameterizer(param_function, sample_sizes, name='sample_size_biased_specific').run_specific(
        pd.DataFrame(np.zeros((10, 5)), columns=[f'feature_{i}' for i in range(dimensions)]),
        pd.DataFrame(np.zeros((10, 1)) + 0.1, columns=['outcome']))


def parameterize_sample_size_general():
    dimensions = 5
    sample_sizes = [{'sample_size': 50},
                    {'sample_size': 100},
                    {'sample_size': 250},
                    {'sample_size': 500},
                    {'sample_size': 750},
                    {'sample_size': 1000},
                    {'sample_size': 1250},
                    {'sample_size': 1500},
                    {'sample_size': 1750},
                    {'sample_size': 2000}
                    ]
    param_function = lambda d: lambda: Experiment() \
        .add_causal_forest(honest=False, min_leaf_size=1, number_of_trees=500) \
        .add_causal_forest(min_leaf_size=1, number_of_trees=500) \
        .add_mean_squared_error() \
        .add_all_effects_generator(dimensions=dimensions, sample_size=d['sample_size'])
    Parameterizer(param_function, sample_sizes, name='sample_size_normal_general').run(save_graphs=True)


def parameterize_number_of_trees():
    dimensions = 5
    tree_numbers = [{'number_of_trees': 2 ** 4},
                    {'number_of_trees': 2 ** 6},
                    {'number_of_trees': 2 ** 8},
                    {'number_of_trees': 2 ** 9},
                    {'number_of_trees': 2 ** 10},
                    {'number_of_trees': 2 ** 11}
                    ]
    param_function = lambda d: lambda: Experiment() \
        .add_causal_forest(honest=False, min_leaf_size=1, number_of_trees=d['number_of_trees']) \
        .add_causal_forest(min_leaf_size=1, number_of_trees=d['number_of_trees']) \
        .add_mean_squared_error() \
        .add_biased_generator(dimensions=dimensions, sample_size=500)
    Parameterizer(param_function, tree_numbers, name='number_of_trees_biased_general').run()
    Parameterizer(param_function, tree_numbers, name='number_of_trees_biased_specific').run_specific(
        pd.DataFrame(np.zeros((40, 5)), columns=[f'feature_{i}' for i in range(dimensions)]),
        pd.DataFrame(np.zeros((40, 1)) + 0.1, columns=['outcome']))


def parameterize_leaf_size():
    dimensions = 5
    leaf_size = [{'min_leaf_size': 1},
                 {'min_leaf_size': 10},
                 {'min_leaf_size': 20},
                 {'min_leaf_size': 50},
                 {'min_leaf_size': 75},
                 {'min_leaf_size': 100}
                 ]
    param_function = lambda d: lambda: Experiment() \
        .add_causal_forest(honest=False, min_leaf_size=d['min_leaf_size'], number_of_trees=500) \
        .add_causal_forest(min_leaf_size=d['min_leaf_size'], number_of_trees=500) \
        .add_mean_squared_error() \
        .add_biased_generator(dimensions=dimensions, sample_size=500)
    Parameterizer(param_function, leaf_size, name='leaf_size_biased_general').run(save_graphs=True)
    Parameterizer(param_function, leaf_size, name='leaf_size_biased_specific').run_specific(
        pd.DataFrame(np.zeros((40, 5)), columns=[f'feature_{i}' for i in range(dimensions)]),
        pd.DataFrame(np.zeros((40, 1)) + 0.1, columns=['outcome']), save_graphs=True)


def parameterize_specific_spiked_sample_size():
    dimensions = 5
    sample_sizes = [{'sample_size': 1000},
                    {'sample_size': 1250},
                    {'sample_size': 1500},
                    {'sample_size': 1750},
                    {'sample_size': 2000},
                    {'sample_size': 2250},
                    {'sample_size': 2500},
                    {'sample_size': 2750},
                    {'sample_size': 3000}
                    ]
    param_function = lambda d: lambda: Experiment() \
        .add_causal_forest(honest=False, min_leaf_size=1, number_of_trees=500) \
        .add_causal_forest(min_leaf_size=1, number_of_trees=500) \
        .add_mean_squared_error() \
        .add_spiked_generator(dimensions=dimensions, sample_size=d['sample_size'])
    Parameterizer(param_function, sample_sizes, name='sample_size_spiked_general').run(save_graphs=True)
    Parameterizer(param_function, sample_sizes, name='sample_size_spiked_specific').run_specific(
        pd.DataFrame(np.ones((40, 5)) / 2, columns=[f'feature_{i}' for i in range(dimensions)]),
        pd.DataFrame(np.zeros((40, 1)) + 15.915494309189528, columns=['outcome']), save_graphs=True)


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


def cevae_v_forest_noise_experiment():
    data_dimensions = 5
    model_dimensions = 9
    sample_size = 10000
    Experiment() \
        .add_cevae(model_dimensions,
                   latent_dim=20,
                   hidden_dim=100,
                   num_layers=3,
                   num_samples=100,
                   outcome_dist="normal") \
        .add_all_metrics() \
        .add_constant_proxied_treatment_effect_generator(data_dimensions, sample_size) \
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


def cevae_easy_synthetic_experiment():
    data_dimensions = 1
    model_dimensions = 1
    sample_sizes = [
        {"proxy_noise": 0.1},
        {"proxy_noise": 0.1},
        # {"proxy_noise": 0.5},
        # {"proxy_noise": 0.1},
        # {"proxy_noise": 0.01},
        # {"proxy_noise": 0.001},
    ]
    param_function = lambda d: lambda: Experiment() \
        .add_cevae(model_dimensions,
                   latent_dim=1,
                   hidden_dim=100,
                   num_layers=3,
                   num_samples=100,
                   batch_size=1000,
                   outcome_dist="normal") \
        .add_all_metrics() \
        .add_easy_generator(data_dimensions, 1000, d['proxy_noise'])
    results = Parameterizer(param_function, sample_sizes, name='cevae_baseline_noise').run(replications=1, save_graphs=False, show_graphs=True)
    print(results)


def cevae_easy_synthetic_bayes():
    data_dimensions = 5
    model_dimensions = 5
    p_bounds = {
        'latent_dim': (1, 20),
        'hidden_dim': (1, 100),
        'num_samples': (100, 1000),
    }

    def opt_funct(latent_dim, hidden_dim, num_samples):
        return Experiment()\
            .add_cevae(model_dimensions,
                       latent_dim=int(latent_dim),
                       hidden_dim=int(hidden_dim),
                       num_layers=2,
                       num_samples=int(num_samples),
                       outcome_dist="normal",
                       batch_size=1000)\
            .add_all_metrics() \
            .add_easy_generator(data_dimensions, 10000) \
            .run() \
            .get_result() * -1
    results = BayesianOptimizer(opt_funct, p_bounds, name='cevae_bayes').run(replications=1)
    print(results)


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


def spiked_experiment():
    dimensions = 5
    sample_size = 5000
    Experiment() \
        .add_causal_forest(honest=False, min_leaf_size=10, number_of_trees=500) \
        .add_causal_forest(min_leaf_size=10, number_of_trees=500) \
        .add_mean_squared_error() \
        .add_spiked_generator(dimensions, sample_size) \
        .run(save_data=True, save_graphs=True, show_graphs=False)


if __name__ == '__main__':
    main()
