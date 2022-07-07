from bayes_opt import BayesianOptimization

from experiment import Experiment
from session import Session
from utils import *


class BayesianOptimizer:
    """
    Tests an experiment with different inputs. All generated data important to this session is stored in the
    'sample/parameterization' directory.
    """

    # def __init__(self, parameter_function: Callable[[Dict[str, float]], Callable[[], Experiment]],
    def __init__(self, parameter_function,
                 param_bounds: Dict[str, Tuple[float, float]], name: str = None):
        """
        Initialization of the Parameterizer class
        :param parameter_function: Function that takes in a dictionary that contains the necessary parameters
        to test and returns a function that outputs a freshly defined experiment based on the parameters formed in the
         dictionary
        :param param_bounds: List of dictionaries to apply on parameter_function
        :param name: name for storing purposes
        """
        self.parameter_function = parameter_function
        self.param_bounds = param_bounds
        os.makedirs('bayesian_optimization', exist_ok=True)
        self.directory = f'bayesian_optimization/params_{len([a for a in os.scandir("bayesian_optimization")]) if name is None else name}'
        os.makedirs(self.directory, exist_ok=True)

    def run(self, replications=1, save_data: bool = True, save_graphs: bool = True, show_graphs: bool = False):
        optimizer = BayesianOptimization(
            f=self.parameter_function,
            pbounds=self.param_bounds,
            verbose=2,
            random_state=1,
        )
        optimizer.maximize(
            init_points=5,
            # init_points=1,
            n_iter=25,
            # n_iter=1,
        )
        print(optimizer.max)

        res = []
        for param in self.param_bounds:
            print(f'Testing parameters {compact_dict_print(param)}')
            experiment_function = self.parameter_function(param)
            session = Session(experiment_function, f'session_{compact_dict_print(param)}')
            results = session.run(replications=replications, save_data=save_data, save_graphs=save_graphs, show_graphs=show_graphs)
            res.append((param, results))
            save_pandas_table(self.directory + f'/table_of_{compact_dict_print(param)}', results)
        if save_graphs:
            self.generate_graphs(res)
        return res

    def run_specific(self, test_set=pd.DataFrame, truth_set=pd.DataFrame, save_graphs: bool = True):
        res = []
        for param in self.param_bounds:
            print(f'Testing parameters {compact_dict_print(param)}')
            experiment_function = self.parameter_function(param)
            session = Session(experiment_function, f'session_{compact_dict_print(param)}')
            results = session.run_specific(test_set, truth_set, save_graphs=save_graphs)
            res.append((param, results))
            save_pandas_table(self.directory + f'/table_of_{compact_dict_print(param)}', results)
        if save_graphs:
            self.generate_graphs(res)

    def generate_graphs(self, res):
        """
        Generates graphs for each parameter that was changed and each metric and model that was tested.
        :param res: Results from the parameterization
        """
        plots = {}
        x = {}
        for params, results in res:
            for key in params:
                if key not in plots:
                    plots[key] = {}
                    x[key] = []
                x[key].append(params[key])
                for model_name, scores in results.iterrows():
                    for score_name in scores.index:
                        new_key = f'{model_name}_{score_name}'
                        if new_key not in plots[key]:
                            plots[key][new_key] = []
                        plots[key][new_key].append(scores[score_name])
        for key in plots:
            plt.clf()
            directory = self.directory + f'/graph_{key}.png'
            plt.title(f'Changing of parameter {key}')
            plt.xlabel(key)
            plt.ylabel('metrics')
            for model in plots[key]:
                plt.plot(x[key], plots[key][model], label=model)
            plt.legend(bbox_to_anchor=(1.04,0), loc="lower left", borderaxespad=0)
            plt.savefig(directory, bbox_inches="tight")
            plt.clf()
