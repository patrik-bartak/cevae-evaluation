"""
In this file all the causal machine learning methods should be defined.
Currently the following Causal methods can be imported from this file:
    * CausalMethod: abstract class represneting the models used for causal ML
    * CausalForest: implementation of causal forests using EconML
    * DragonNet: DragonNet implementation defined by https://github.com/claudiashi57/dragonnet
"""
import tensorflow as tf
import torch
from typing import Dict

from other_methods.dragonnet.experiment.models import regression_loss, binary_classification_loss, \
    treatment_accuracy, track_epsilon

from pyro.contrib.cevae import CEVAE

tf.compat.v1.disable_eager_execution()
import pandas as pd
from econml.dml import CausalForestDML as EconCausalForest
from abc import abstractmethod, ABC
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, TerminateOnNaN
from keras.optimizer_v1 import Adam, SGD
from other_methods.dragonnet.experiment.ihdp_main import make_dragonnet
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
    def train(self, x, y, w, x_test=None, y_test=None, t_test=None, ite_test_truth=None, ite_train_truth=None):
        """
        Method that trains the model with necessary data.
        :param x: List of feature vectors
        :param y: List of outcomes
        :param w: List of treatments
        :param x_test: Test feature vectors to compute validation loss
        :param y_test: Test outcomes to compute validation loss
        :param t_test: Test treatments to compute validation loss
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
        ).to(self.device)

    def __str__(self):
        return f'cevae_{self.id}'

    def __init__(self,
                 feature_dim, outcome_dist, model_dist, latent_dim,
                 hidden_dim, num_layers, num_samples, batch_size,
                 num_epochs,
                 learning_rate,
                 learning_rate_decay,
                 weight_decay,
                 id: int = 0, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        print("Setting default tensor type")
        if device == torch.device("cuda"):
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
        else:
            torch.set_default_tensor_type(torch.FloatTensor)
        print("Finished setting tensor type")
        self.device = device
        self.batch_size = batch_size
        print(f"Using {self.device}")
        self.cevae = CEVAE(
            feature_dim, outcome_dist, latent_dim, hidden_dim, num_layers, num_samples, latent_dist=model_dist
        ).to(self.device).train(True)
        # self.cevae = CEVAE(
        #     feature_dim, outcome_dist, latent_dim, hidden_dim, num_layers, num_samples, latent_dist="bernoulli"
        # ).to(self.device).train(True)
        self.config = dict(
            feature_dim=feature_dim,
            outcome_dist=outcome_dist,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_samples=num_samples,
        )
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.weight_decay = weight_decay
        self.id = id

    def train(self, x_train, y_train, t_train, x_test=None, y_test=None, t_test=None,
              ite_test_truth=None, ite_train_truth=None,
              log_every=100) -> Dict[str, List[float]]:
        x_train_tensor = torch.FloatTensor(x_train.values)
        y_train_tensor = torch.FloatTensor(y_train.values)
        t_train_tensor = torch.FloatTensor(t_train.values)
        if x_test is not None:
            print("Using validation loss")
            x_test_tensor = torch.FloatTensor(x_test.values)
            y_test_tensor = torch.FloatTensor(y_test.values)
            t_test_tensor = torch.FloatTensor(t_test.values)
        else:
            x_test_tensor = None
            y_test_tensor = None
            t_test_tensor = None
        data_num_features = x_train_tensor.size(-1)
        model_num_dims = self.config['feature_dim']
        assert data_num_features == model_num_dims, f"Expected data to have {model_num_dims} features, but found {data_num_features}"
        print("X tensor size:", x_train_tensor.size())
        print("y tensor size:", y_train_tensor.size())
        print("t tensor size:", t_train_tensor.size())
        self.cevae.train(True)
        print("Training: ", self.cevae.training)
        # self.dummy.fit(x_train_tensor.to(self.device), t_train_tensor.to(self.device), y_train_tensor.to(self.device),
        #                               x_test_tensor.to(self.device), t_test_tensor.to(self.device), y_test_tensor.to(self.device),
        #                               ite_truth,
        #                               num_epochs, self.batch_size, learning_rate,
        #                               learning_rate_decay, weight_decay, log_every)
        # del self.dummy
        # gc.collect()
        epoch_losses = self.cevae.fit_new(x_train_tensor.to(self.device), t_train_tensor.to(self.device), y_train_tensor.to(self.device),
                                          x_test_tensor.to(self.device), t_test_tensor.to(self.device), y_test_tensor.to(self.device),
                                          ite_test_truth, ite_train_truth,
                                          self.num_epochs, self.batch_size, self.learning_rate,
                                          self.learning_rate_decay, self.weight_decay, log_every)
        print("Fitting model hash: ", hash(self))
        return epoch_losses

    def estimate_causal_effect(self, x_test):
        x_tensor = torch.FloatTensor(x_test.values)
        print("X test tensor size:", x_tensor.size())
        self.cevae.train(False)
        ite = self.cevae.ite(x_tensor.to(self.device)).cpu()  # individual treatment effect
        print("ITE tensor size:", ite.size())
        print("Estimating model hash: ", hash(self))
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

    def train(self, x, y, w, x_test=None, y_test=None, t_test=None, ite_truth=None):
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

    def train(self, x, y, w, x_test=None, y_test=None, t_test=None):
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


class DummyMethod(CausalMethod):

    def __init__(self, id: int = 0):
        self.num_times_trained = 0
        self.truth = None
        self.id = id

    def estimate_causal_effect(self, x):
        return np.ones(np.shape(x)[0]) * self.num_times_trained

    def train(self, x, y, w, x_test=None, y_test=None, t_test=None, ite_truth=None):
        self.truth = ite_truth
        self.num_times_trained += 1
        return {}

    def create_training_truth(self, outcome, main_effect, treatment_effect, treatment_propensity,
                              y0, y1, noise, cate):
        return outcome

    def create_testing_truth(self, outcome, main_effect, treatment_effect, treatment_propensity, y0,
                             y1, noise, cate):
        return treatment_effect

    def reset(self):
        pass

    def __str__(self):
        return f'dummy_{self.id}'


class CausalEffectVariationalAutoencoderOriginal(CausalMethod):

    def create_training_truth(self, outcome, main_effect, treatment_effect, treatment_propensity,
                              y0, y1, noise, cate):
        return outcome

    def create_testing_truth(self, outcome, main_effect, treatment_effect, treatment_propensity, y0,
                             y1, noise, cate):
        return treatment_effect

    def reset(self):
        pass
        # self.cevae = CEVAE(
        #     self.config["feature_dim"], self.config["outcome_dist"],
        #     self.config["latent_dim"], self.config["hidden_dim"],
        #     self.config["num_layers"], self.config["num_samples"]
        # ).to(self.device)

    def __str__(self):
        return f'cevae_original_{self.id}'

    def __init__(self,
                 feature_dim, outcome_dist, latent_dim,
                 hidden_dim, num_layers, num_samples, batch_size,
                 id: int = 0, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        # if device == torch.device("cuda"):
        #     torch.set_default_tensor_type(torch.cuda.FloatTensor)
        # else:
        #     torch.set_default_tensor_type(torch.FloatTensor)
        self.device = device
        self.batch_size = batch_size
        print(f"Using {self.device}")
        self.config = dict(
            feature_dim=feature_dim,
            outcome_dist=outcome_dist,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_samples=num_samples,
        )
        self.id = id

    def train(self, x_train, y_train, t_train, x_test=None, y_test=None, t_test=None, ite_truth=None,
              num_epochs=100,
              learning_rate=1e-3,
              learning_rate_decay=0.1,
              weight_decay=1e-4,
              log_every=100) -> Dict[str, List[float]]:
        # Requirements
        # Edward 1.3.1
        # Tensorflow 1.1.0
        # Progressbar 2.3
        # Scikit-learn 0.18.1
        import os
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        # os.add_dll_directory("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\cudnn-8.0-windows10-v5.1\\bin")
        import tensorflow.compat.v1 as tf
        tf.disable_v2_behavior()
        print(tf.__version__)
        import edward as ed
        from edward.models import Bernoulli, Normal
        from progressbar import ETA, Bar, Percentage, ProgressBar
        from src.data.ihdp import IHDP
        from evaluation import Evaluator
        import numpy as np
        import time
        from scipy.stats import sem
        from utils import fc_net, get_y0_y1
        from argparse import ArgumentParser
        parser = ArgumentParser()
        parser.add_argument('-reps', type=int, default=10)
        parser.add_argument('-earl', type=int, default=10)
        parser.add_argument('-lr', type=float, default=0.001)
        parser.add_argument('-opt', choices=['adam', 'adamax'], default='adam')
        parser.add_argument('-epochs', type=int, default=100)
        parser.add_argument('-print_every', type=int, default=10)
        args = parser.parse_args()

        args.true_post = True

        dataset = IHDP(replications=args.reps)
        dimx = 25
        scores = np.zeros((args.reps, 3))
        scores_test = np.zeros((args.reps, 3))

        M = None  # batch size during training
        d = 20  # latent dimension
        lamba = 1e-4  # weight decay
        nh, h = 3, 200  # number and size of hidden layers

        for i, (train, valid, test, contfeats, binfeats) in enumerate(dataset.get_train_valid_test()):
            print('\nReplication {}/{}'.format(i + 1, args.reps))
            (xtr, ttr, ytr), (y_cftr, mu0tr, mu1tr) = train
            (xva, tva, yva), (y_cfva, mu0va, mu1va) = valid
            (xte, tte, yte), (y_cfte, mu0te, mu1te) = test
            evaluator_test = Evaluator(yte, tte, y_cf=y_cfte, mu0=mu0te, mu1=mu1te)

            # reorder features with binary first and continuous after
            perm = binfeats + contfeats
            xtr, xva, xte = xtr[:, perm], xva[:, perm], xte[:, perm]

            xalltr, talltr, yalltr = np.concatenate([xtr, xva], axis=0), np.concatenate([ttr, tva],
                                                                                        axis=0), np.concatenate(
                [ytr, yva], axis=0)
            evaluator_train = Evaluator(yalltr, talltr, y_cf=np.concatenate([y_cftr, y_cfva], axis=0),
                                        mu0=np.concatenate([mu0tr, mu0va], axis=0),
                                        mu1=np.concatenate([mu1tr, mu1va], axis=0))

            # zero mean, unit variance for y during training
            ym, ys = np.mean(ytr), np.std(ytr)
            ytr, yva = (ytr - ym) / ys, (yva - ym) / ys
            best_logpvalid = - np.inf

            with tf.Graph().as_default():
                sess = tf.InteractiveSession()

                ed.set_seed(1)
                np.random.seed(1)
                tf.set_random_seed(1)

                x_ph_bin = tf.placeholder(tf.float32, [M, len(binfeats)], name='x_bin')  # binary inputs
                x_ph_cont = tf.placeholder(tf.float32, [M, len(contfeats)],
                                           name='x_cont')  # continuous inputs
                t_ph = tf.placeholder(tf.float32, [M, 1])
                y_ph = tf.placeholder(tf.float32, [M, 1])

                x_ph = tf.concat([x_ph_bin, x_ph_cont], 1)
                activation = tf.nn.elu

                # CEVAE model (decoder)
                # p(z)
                z = Normal(loc=tf.zeros([tf.shape(x_ph)[0], d]), scale=tf.ones([tf.shape(x_ph)[0], d]))

                # p(x|z)
                hx = fc_net(z, (nh - 1) * [h], [], 'px_z_shared', lamba=lamba, activation=activation)
                logits = fc_net(hx, [h], [[len(binfeats), None]], 'px_z_bin'.format(i + 1), lamba=lamba,
                                activation=activation)
                x1 = Bernoulli(logits=logits, dtype=tf.float32, name='bernoulli_px_z')

                mu, sigma = fc_net(hx, [h], [[len(contfeats), None], [len(contfeats), tf.nn.softplus]],
                                   'px_z_cont', lamba=lamba,
                                   activation=activation)
                x2 = Normal(loc=mu, scale=sigma, name='gaussian_px_z')

                # p(t|z)
                logits = fc_net(z, [h], [[1, None]], 'pt_z', lamba=lamba, activation=activation)
                t = Bernoulli(logits=logits, dtype=tf.float32)

                # p(y|t,z)
                mu2_t0 = fc_net(z, nh * [h], [[1, None]], 'py_t0z', lamba=lamba, activation=activation)
                mu2_t1 = fc_net(z, nh * [h], [[1, None]], 'py_t1z', lamba=lamba, activation=activation)
                y = Normal(loc=t * mu2_t1 + (1. - t) * mu2_t0, scale=tf.ones_like(mu2_t0))

                # CEVAE variational approximation (encoder)
                # q(t|x)
                logits_t = fc_net(x_ph, [d], [[1, None]], 'qt', lamba=lamba, activation=activation)
                qt = Bernoulli(logits=logits_t, dtype=tf.float32)
                # q(y|x,t)
                hqy = fc_net(x_ph, (nh - 1) * [h], [], 'qy_xt_shared', lamba=lamba, activation=activation)
                mu_qy_t0 = fc_net(hqy, [h], [[1, None]], 'qy_xt0', lamba=lamba, activation=activation)
                mu_qy_t1 = fc_net(hqy, [h], [[1, None]], 'qy_xt1', lamba=lamba, activation=activation)
                qy = Normal(loc=qt * mu_qy_t1 + (1. - qt) * mu_qy_t0, scale=tf.ones_like(mu_qy_t0))
                # q(z|x,t,y)
                inpt2 = tf.concat([x_ph, qy], 1)
                hqz = fc_net(inpt2, (nh - 1) * [h], [], 'qz_xty_shared', lamba=lamba, activation=activation)
                muq_t0, sigmaq_t0 = fc_net(hqz, [h], [[d, None], [d, tf.nn.softplus]], 'qz_xt0',
                                           lamba=lamba,
                                           activation=activation)
                muq_t1, sigmaq_t1 = fc_net(hqz, [h], [[d, None], [d, tf.nn.softplus]], 'qz_xt1',
                                           lamba=lamba,
                                           activation=activation)
                qz = Normal(loc=qt * muq_t1 + (1. - qt) * muq_t0,
                            scale=qt * sigmaq_t1 + (1. - qt) * sigmaq_t0)

                # Create data dictionary for edward
                data = {x1: x_ph_bin, x2: x_ph_cont, y: y_ph, qt: t_ph, t: t_ph, qy: y_ph}

                # sample posterior predictive for p(y|z,t)
                y_post = ed.copy(y, {z: qz, t: t_ph}, scope='y_post')
                # crude approximation of the above
                y_post_mean = ed.copy(y, {z: qz.mean(), t: t_ph}, scope='y_post_mean')
                # construct a deterministic version (i.e. use the mean of the approximate posterior) of the lower bound
                # for early stopping according to a validation set
                y_post_eval = ed.copy(y, {z: qz.mean(), qt: t_ph, qy: y_ph, t: t_ph}, scope='y_post_eval')
                x1_post_eval = ed.copy(x1, {z: qz.mean(), qt: t_ph, qy: y_ph}, scope='x1_post_eval')
                x2_post_eval = ed.copy(x2, {z: qz.mean(), qt: t_ph, qy: y_ph}, scope='x2_post_eval')
                t_post_eval = ed.copy(t, {z: qz.mean(), qt: t_ph, qy: y_ph}, scope='t_post_eval')
                logp_valid = tf.reduce_mean(
                    tf.reduce_sum(y_post_eval.log_prob(y_ph) + t_post_eval.log_prob(t_ph), axis=1) +
                    tf.reduce_sum(x1_post_eval.log_prob(x_ph_bin), axis=1) +
                    tf.reduce_sum(x2_post_eval.log_prob(x_ph_cont), axis=1) +
                    tf.reduce_sum(z.log_prob(qz.mean()) - qz.log_prob(qz.mean()), axis=1))

                inference = ed.KLqp({z: qz}, data)
                optimizer = tf.train.AdamOptimizer(learning_rate=args.lr)
                inference.initialize(optimizer=optimizer)

                saver = tf.train.Saver(tf.contrib.slim.get_variables())
                tf.global_variables_initializer().run()

                n_epoch, n_iter_per_epoch, idx = args.epochs, 10 * int(xtr.shape[0] / 100), np.arange(
                    xtr.shape[0])

                # dictionaries needed for evaluation
                tr0, tr1 = np.zeros((xalltr.shape[0], 1)), np.ones((xalltr.shape[0], 1))
                tr0t, tr1t = np.zeros((xte.shape[0], 1)), np.ones((xte.shape[0], 1))
                f1 = {x_ph_bin: xalltr[:, 0:len(binfeats)], x_ph_cont: xalltr[:, len(binfeats):], t_ph: tr1}
                f0 = {x_ph_bin: xalltr[:, 0:len(binfeats)], x_ph_cont: xalltr[:, len(binfeats):], t_ph: tr0}
                f1t = {x_ph_bin: xte[:, 0:len(binfeats)], x_ph_cont: xte[:, len(binfeats):], t_ph: tr1t}
                f0t = {x_ph_bin: xte[:, 0:len(binfeats)], x_ph_cont: xte[:, len(binfeats):], t_ph: tr0t}

                for epoch in range(n_epoch):
                    avg_loss = 0.0

                    t0 = time.time()
                    widgets = ["epoch #%d|" % epoch, Percentage(), Bar(), ETA()]
                    pbar = ProgressBar(n_iter_per_epoch, widgets=widgets)
                    pbar.start()
                    np.random.shuffle(idx)
                    for j in range(n_iter_per_epoch):
                        pbar.update(j)
                        batch = np.random.choice(idx, 100)
                        x_train, y_train, t_train = xtr[batch], ytr[batch], ttr[batch]
                        info_dict = inference.update(feed_dict={x_ph_bin: x_train[:, 0:len(binfeats)],
                                                                x_ph_cont: x_train[:, len(binfeats):],
                                                                t_ph: t_train, y_ph: y_train})
                        avg_loss += info_dict['loss']

                    avg_loss = avg_loss / n_iter_per_epoch
                    avg_loss = avg_loss / 100

                    if epoch % args.earl == 0 or epoch == (n_epoch - 1):
                        logpvalid = sess.run(logp_valid, feed_dict={x_ph_bin: xva[:, 0:len(binfeats)],
                                                                    x_ph_cont: xva[:, len(binfeats):],
                                                                    t_ph: tva, y_ph: yva})
                        if logpvalid >= best_logpvalid:
                            print('Improved validation bound, old: {:0.3f}, new: {:0.3f}'.format(
                                best_logpvalid, logpvalid))
                            best_logpvalid = logpvalid
                            saver.save(sess, 'models/m6-ihdp')

                    if epoch % args.print_every == 0:
                        y0, y1 = get_y0_y1(sess, y_post, f0, f1, shape=yalltr.shape, L=1)
                        y0, y1 = y0 * ys + ym, y1 * ys + ym
                        score_train = evaluator_train.calc_stats(y1, y0)
                        rmses_train = evaluator_train.y_errors(y0, y1)

                        y0, y1 = get_y0_y1(sess, y_post, f0t, f1t, shape=yte.shape, L=1)
                        y0, y1 = y0 * ys + ym, y1 * ys + ym
                        score_test = evaluator_test.calc_stats(y1, y0)

                        print(
                            "Epoch: {}/{}, log p(x) >= {:0.3f}, ite_tr: {:0.3f}, ate_tr: {:0.3f}, pehe_tr: {:0.3f}, " \
                            "rmse_f_tr: {:0.3f}, rmse_cf_tr: {:0.3f}, ite_te: {:0.3f}, ate_te: {:0.3f}, pehe_te: {:0.3f}, " \
                            "dt: {:0.3f}".format(epoch + 1, n_epoch, avg_loss, score_train[0],
                                                 score_train[1], score_train[2],
                                                 rmses_train[0], rmses_train[1], score_test[0],
                                                 score_test[1], score_test[2],
                                                 time.time() - t0))

                saver.restore(sess, 'models/m6-ihdp')
                y0, y1 = get_y0_y1(sess, y_post, f0, f1, shape=yalltr.shape, L=100)
                y0, y1 = y0 * ys + ym, y1 * ys + ym
                score = evaluator_train.calc_stats(y1, y0)
                scores[i, :] = score

                y0t, y1t = get_y0_y1(sess, y_post, f0t, f1t, shape=yte.shape, L=100)
                y0t, y1t = y0t * ys + ym, y1t * ys + ym
                score_test = evaluator_test.calc_stats(y1t, y0t)
                scores_test[i, :] = score_test

                print('Replication: {}/{}, tr_ite: {:0.3f}, tr_ate: {:0.3f}, tr_pehe: {:0.3f}' \
                      ', te_ite: {:0.3f}, te_ate: {:0.3f}, te_pehe: {:0.3f}'.format(i + 1, args.reps,
                                                                                    score[0], score[1],
                                                                                    score[2],
                                                                                    score_test[0],
                                                                                    score_test[1],
                                                                                    score_test[2]))
                sess.close()

        print('CEVAE model total scores')
        means, stds = np.mean(scores, axis=0), sem(scores, axis=0)
        print('train ITE: {:.3f}+-{:.3f}, train ATE: {:.3f}+-{:.3f}, train PEHE: {:.3f}+-{:.3f}' \
              ''.format(means[0], stds[0], means[1], stds[1], means[2], stds[2]))

        means, stds = np.mean(scores_test, axis=0), sem(scores_test, axis=0)
        print('test ITE: {:.3f}+-{:.3f}, test ATE: {:.3f}+-{:.3f}, test PEHE: {:.3f}+-{:.3f}' \
              ''.format(means[0], stds[0], means[1], stds[1], means[2], stds[2]))

    def estimate_causal_effect(self, x_test):
        x_tensor = torch.FloatTensor(x_test.values)
        print("X test tensor size:", x_tensor.size())
        self.cevae.train(False)
        ite = self.cevae.ite(x_tensor.to(self.device)).cpu()  # individual treatment effect
        print("ITE tensor size:", ite.size())
        print("Estimating model hash: ", hash(self))
        # ate = ite.mean()  # average treatment effect
        return ite.numpy()
