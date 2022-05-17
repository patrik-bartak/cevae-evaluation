"""Comparator

This file is used to defined functions that runs some models on one set of data and measures some metrics on their
performances.

This file can also be imported as a module and contains the following
functions and classes:

    * run: Method that runs input models and measures some metrics. It outputs a dataframe containing the results.
    * run_model: Method that runs one specific model on the input data and input metrics.
"""

from causal_effect_methods import *
from data_generator import *
from sklearn.model_selection import train_test_split
from utils import save_pandas_table


def run(methods: Dict[str, CausalMethod],
        score_functions: Dict[str, Callable[[List[float], List[float]], float]],
        data_generator: Generator = None, data_file: str = None, samples: int = 500, save_table: bool = False,
        dir: str = '', show_graphs=False, save_graphs=False):
    """
    Method that runs input models and measures some metrics. It outputs a dataframe containing the results.
    If a data generator is defined, the method will use the generator. Make sure only generator OR data file is defined.
    :param methods: dictionary of all methods that are to be run
    :param score_functions: dictionary of all metrics that are to be observed
    :param data_generator: generator of data that can be used to generate data
    :param data_file: file from which data will be read from
    :param samples: number of samples to generate if generator is used
    :param save_table: boolean representing whether all data should be stored
    :param dir: directory of where to store data
    :return: dataframe representing the results observed. Each row is a model where each column is a metric.
    """
    assert data_generator is not None or data_file is not None, "Data must be either generated or read from a file."
    scoring_list = [score_functions[key] for key in score_functions]
    columns = [key for key in score_functions.keys()]
    columns.insert(0, 'method_name')
    # X, W, y, main_effect, true_effect, propensity, y0, y1, noise, cate = None, None, None, None, None,\
    #                                                                      None, None, None, None, None
    if data_generator is not None:
        generated_data = load_data_from_generator(data_generator, samples)
        if len(generated_data) == 10:
            X, W, y, main_effect, true_effect, propensity, y0, y1, noise, cate = generated_data
        else:
            X, proxies, W, y, main_effect, true_effect, propensity, y0, y1, noise, cate = generated_data

    elif data_file is not None:
        loaded_data = load_data_from_file(data_file)
        if len(loaded_data) == 10:
            print("Using non-proxy data")
            X, W, y, main_effect, true_effect, propensity, y0, y1, noise, cate = loaded_data
        else:
            print("Using proxy data")
            X, proxies, W, y, main_effect, true_effect, propensity, y0, y1, noise, cate = loaded_data

    df = pd.DataFrame([], columns=columns)
    all_data = X.join(proxies).join([W, y, main_effect, true_effect, propensity, y0, y1, noise, cate])
    for method in methods:
        model = methods[method]
        # results = run_model(model, scoring_list, X, W, y, main_effect, true_effect, propensity, y0, y1, noise, cate,
        #                     save_table=save_table, dir=dir)
        results = run_model(model, scoring_list, proxies, W, y, main_effect, true_effect, propensity, y0, y1, noise, cate,
                            save_table=save_table, dir=dir)
        results.insert(0, method)
        df.loc[len(df.index)] = results
        if show_graphs or save_graphs:
            grapher = Grapher(dir, show_graphs, save_graphs)
            inferred_treatment_effect = pd.DataFrame(data=model.estimate_causal_effect(select_proxies(all_data)))

            fn = f'/{model}_estimation_feat0'
            grapher.scatter_2d(fn, all_data['feature_0'], inferred_treatment_effect,
                               "Feature 0", "Estimated Treatment Effect")
            fn = f'/{model}_estimation_2d_feat01'
            grapher.scatter_2d_color(fn, all_data['feature_0'], all_data['feature_1'], inferred_treatment_effect,
                                     "Feature 0", "Feature 1", "Estimated Treatment Effect Strength")

    df = df.set_index('method_name')
    if save_table:
        save_pandas_table(dir + '/inter_table', df)
    return df

def run_model(model: CausalMethod, score_functions: List[Callable[[List[float], List[float]], float]],
              feature_data: pd.DataFrame, treatment: pd.DataFrame, outcome: pd.DataFrame, main_effect: pd.DataFrame,
              treatment_effect: pd.DataFrame, treatment_propensity: pd.DataFrame,
              y0: pd.DataFrame, y1: pd.DataFrame, noise: pd.DataFrame, cate: pd.DataFrame,
              save_table=False, dir=''):
    """
    Method that runs one specific model on the input data and input metrics.
    :param model: Model to test out
    :param score_functions: list of metrics to observe
    :param feature_data: dataframe containing all the features
    :param treatment: dataframe containing the treatments
    :param outcome: dataframe containing the outcomes
    :param main_effect: dataframe containing the main_effects
    :param treatment_effect: dataframe containing the treatment effects
    :param treatment_propensity: dataframe containing the treatment propensities
    :param y0: dataframe containing y0's
    :param y1: dataframe containing y1's
    :param noise: dataframe containing noise
    :param cate: dataframe containing cate
    :param save_table: boolean representing whether table of predictions and the base truth should be stored
    :param dir: directory where to save table
    :return: list of observed metrics
    """
    all_data = feature_data.join(treatment)
    all_data = all_data.join(outcome)
    all_data = all_data.join(main_effect)
    all_data = all_data.join(treatment_effect)
    all_data = all_data.join(treatment_propensity)
    all_data = all_data.join(y0)
    all_data = all_data.join(y1)
    all_data = all_data.join(noise)
    all_data = all_data.join(cate)
    # Ensure that X_train and X_test hold all values needed
    X_train, X_test, y_train, y_test = train_test_split(all_data,
                                                        model.create_training_truth(outcome, main_effect,
                                                                                    treatment_effect,
                                                                                    treatment_propensity,
                                                                                    y0, y1, noise, cate),
                                                        test_size=0.25, random_state=42)
    # Select only features for training
    # model.train(select_features(X_train), y_train, X_train['treatment'])
    model.train(select_proxies(X_train), y_train, X_train['treatment'])

    # Overwrite y_test based on the model prediction expectation
    y_test = model.create_testing_truth(X_test['outcome'], X_test['main_effect'], X_test['treatment_effect'],
                                        X_test['propensity'], X_test['y0'], X_test['y1'], X_test['noise'], X_test['cate'])

    # Select only features for testing
    # results = model.estimate_causal_effect(select_features(X_test))
    results = model.estimate_causal_effect(select_proxies(X_test))
    if save_table:
        select_proxies(X_test).to_csv(dir + f'/testing_set_{model}.csv')
        y_test.to_csv(dir + f'/base_truth_for_testing_set_{model}.csv')
        save_pandas_table(dir + f'/table_predictions_{model}', pd.DataFrame(
            results,
            columns=[
                f'prediction_{i}' for i in range(results.shape[1])
            ] if len(results.shape) > 1 else ['prediction']
        ))
    return [score_function(y_test.to_numpy(), results) for score_function in score_functions]
