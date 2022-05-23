"""
In this file all the causal machine learning methods should be defined.
Currently the following Causal methods can be imported from this file:
    * CausalMethod: abstract class represneting the models used for causal ML
    * CausalForest: implementation of causal forests using EconML
    * DragonNet: DragonNet implementation defined by https://github.com/claudiashi57/dragonnet
"""
import math
import numpy as np

import tensorflow as tf
import torch

from sample.other_methods.dragonnet.experiment.models import regression_loss, binary_classification_loss, \
    treatment_accuracy, track_epsilon

from pyro.contrib.cevae import CEVAE

tf.compat.v1.disable_eager_execution()
from econml.dml import CausalForestDML as EconCausalForest
from abc import abstractmethod, ABC
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, TerminateOnNaN
from keras.optimizer_v1 import Adam, SGD
from load_data import *
from sample.other_methods.dragonnet.experiment.ihdp_main import make_dragonnet
from keras.metrics import *


class CausalMethod(ABC):
    """
    Abstract class representing all CML models.
    """

    @abstractmethod
    def estimate_causal_effect(self, x):
        """
        Estimates the causal effect.
        :param x: feature vector with all necessary features
        :return: estimation of the output of the model
        """
        pass

    @abstractmethod
    def train(self, x, y, w):
        """
        Method that trains the model with necessary data.
        :param x: List of feature vectors
        :param y: List of outcomes
        :param w: List of treatments
        """
        pass

    @abstractmethod
    def create_training_truth(self, outcome, main_effect, treatment_effect, treatment_propensity, y0, y1, noise, cate):
        """
        Creates the outcomes for training, as some models might require a different training data.
        :param outcome: outcome Y
        :param main_effect: effect of X on Y
        :param treatment_effect: effect of W on Y
        :param treatment_propensity: effect of X on W
        :param y0: outcome with no treatment
        :param y1: outcome with treatment
        :param noise: noise of the sample
        :param cate: cate of the sample
        :return: data used for training
        """
        pass

    @abstractmethod
    def create_testing_truth(self, outcome, main_effect, treatment_effect, treatment_propensity, y0, y1, noise, cate):
        """
        Creates the outcomes for testing, as some models might require a different testing data.
        :param outcome: outcome Y
        :param main_effect: effect of X on Y
        :param treatment_effect: effect of W on Y
        :param treatment_propensity: effect of X on W
        :param y0: outcome with no treatment
        :param y1: outcome with treatment
        :param noise: noise of the sample
        :param cate: cate of the sample
        :return: data used for testing
        """
        pass

    @abstractmethod
    def reset(self):
        """
        Resets the model to an untrained stage.
        """
        pass

    @abstractmethod
    def __str__(self):
        """
        Constructs a string that characterizes the model.
        :return: string representing the model
        """
        pass


class CausalEffectVariationalAutoencoder(CausalMethod):

    def create_training_truth(self, outcome, main_effect, treatment_effect, treatment_propensity,
                              y0, y1, noise, cate):
        return outcome

    def create_testing_truth(self, outcome, main_effect, treatment_effect, treatment_propensity, y0,
                             y1, noise, cate):
        return treatment_effect

    def reset(self):
        self.cevae = CEVAE(
            self.config["feature_dim"], self.config["outcome_dist"],
            self.config["latent_dim"], self.config["hidden_dim"],
            self.config["num_layers"], self.config["num_samples"]
        )

    def __str__(self):
        return f'cevae_{self.id}'

    def __init__(self,
                 feature_dim, outcome_dist, latent_dim,
                 hidden_dim, num_layers, num_samples,
                 id: int = 0, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        if device == torch.device("cuda"):
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
        else:
            torch.set_default_tensor_type(torch.FloatTensor)
        self.device = device
        print(f"Using {self.device}")
        self.cevae = CEVAE(
            feature_dim, outcome_dist, latent_dim, hidden_dim, num_layers, num_samples
        ).to(self.device)
        self.config = dict(
            feature_dim=feature_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_samples=num_samples,
        )
        self.id = id

    def train(self, x_train, y_train, t_train,
              num_epochs=100,
              batch_size=100,
              learning_rate=1e-3,
              learning_rate_decay=0.1,
              weight_decay=1e-4,
              log_every=100):
        x_train_tensor = torch.FloatTensor(x_train.values)
        y_train_tensor = torch.FloatTensor(y_train.values)
        t_train_tensor = torch.FloatTensor(t_train.values)
        data_num_features = x_train_tensor.size(-1)
        model_num_dims = self.config['feature_dim']
        assert data_num_features == model_num_dims, f"Expected data to have {model_num_dims} features, but found {data_num_features}"
        print("X tensor size:", x_train_tensor.size())
        print("y tensor size:", y_train_tensor.size())
        print("t tensor size:", t_train_tensor.size())
        batch_losses = self.cevae.fit(x_train_tensor.to(self.device), t_train_tensor.to(self.device), y_train_tensor.to(self.device),
                       num_epochs, batch_size, learning_rate,
                       learning_rate_decay, weight_decay, log_every)
        num_batches = int(math.ceil(float(x_train_tensor.size(0)) / float(batch_size)))
        epoch_losses = np.mean(np.reshape(batch_losses, (-1, num_batches)), axis=1)
        return epoch_losses

    def estimate_causal_effect(self, x_test):
        x_tensor = torch.FloatTensor(x_test.values)
        print("X test tensor size:", x_tensor.size())
        ite = self.cevae.ite(x_tensor.to(self.device)).cpu()  # individual treatment effect
        print("ITE tensor size:", ite.size())
        # ate = ite.mean()  # average treatment effect
        return ite.numpy()


class CausalForest(CausalMethod):

    def __init__(self, number_of_trees, method_effect='auto', method_predict='auto', k=1, honest: bool = True, id: int = 0):
        self.forest = EconCausalForest(model_t=method_effect, model_y=method_predict, n_estimators=number_of_trees,
                                       min_samples_leaf=k, criterion='mse', random_state=42, honest=honest)
        self.id = id

    def reset(self):
        self.forest = EconCausalForest(model_t=self.forest.model_t, model_y=self.forest.model_y,
                                       n_estimators=self.forest.n_estimators,
                                       min_samples_leaf=self.forest.min_samples_leaf, criterion=self.forest.criterion,
                                       random_state=self.forest.random_state, honest=self.forest.honest)

    def train(self, x, y, w):
        self.forest.fit(Y=y,
                        T=w,
                        X=x,
                        cache_values=True)

    def estimate_causal_effect(self, x):
        return self.forest.effect(x)

    def create_training_truth(self, outcome, main_effect, treatment_effect, treatment_propensity, y0, y1, noise, cate):
        return outcome

    def create_testing_truth(self, outcome, main_effect, treatment_effect, treatment_propensity, y0, y1, noise, cate):
        return cate

    def __str__(self):
        return f'causal_forest_{self.id}'


class DragonNet(CausalMethod):

    # Not sure what reg_l2 is but I base it on DragonNet implementation
    def __init__(self, dimensions, reg_l2=0.01, id: int = 0):
        self.dimensions: int = dimensions
        self.reg_l2: float = reg_l2
        self.dragonnet = make_dragonnet(dimensions, reg_l2)
        self.id = id

    def reset(self):
        self.dragonnet = make_dragonnet(self.dimensions, self.reg_l2)

    def train(self, x, y, w):
        metrics = [regression_loss, binary_classification_loss, treatment_accuracy, track_epsilon]

        self.dragonnet.compile(
            optimizer=Adam(lr=1e-3),
            loss=mean_squared_error, metrics=metrics)

        adam_callbacks = [
            TerminateOnNaN(),
            EarlyStopping(monitor='val_loss', patience=2, min_delta=0.),
            ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, verbose=0, mode='auto',
                              min_delta=1e-8, cooldown=0, min_lr=0)

        ]

        self.dragonnet.fit(x=x, y=y, callbacks=adam_callbacks,
                      validation_split=0.2,
                      epochs=100,
                      batch_size=64, verbose=0)

        sgd_callbacks = [
            TerminateOnNaN(),
            EarlyStopping(monitor='val_loss', patience=40, min_delta=0.),
            ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, verbose=0, mode='auto',
                              min_delta=0., cooldown=0, min_lr=0)
        ]

        sgd_lr = 1e-5
        momentum = 0.9
        self.dragonnet.compile(optimizer=SGD(lr=sgd_lr, momentum=momentum, nesterov=True), loss=mean_squared_error,
                          metrics=metrics)
        self.dragonnet.fit(x=x, y=y, callbacks=sgd_callbacks,
                      validation_split=0.5,
                      epochs=300,
                      batch_size=64, verbose=0)

    def estimate_causal_effect(self, x):
        results = self.dragonnet.predict(x)
        return results

    def create_training_truth(self, outcome, main_effect, treatment_effect, treatment_propensity, y0, y1, noise, cate):
        base_truth = pd.DataFrame(y0).join(y1)
        base_truth = base_truth.join(treatment_propensity)
        base_truth = base_truth.join(noise)
        return base_truth

    def create_testing_truth(self, outcome, main_effect, treatment_effect, treatment_propensity, y0, y1, noise, cate):
        return self.create_training_truth(outcome, main_effect, treatment_effect, treatment_propensity, y0, y1, noise, cate)

    def __str__(self):
        return f'dragonnet_{self.id}'
