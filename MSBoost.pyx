# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: initializedcheck=False
# cython: cdivision=True
# cython: infer_types=True
# cython: profile=False
# cython: binding=False
# cython: optimize.unpack_method_calls=True
# cython: optimize.use_switch=True
# cython: embedsignature=False
# cython: overflowcheck=False  
# cython: autotestdict=False  
# cython: linetrace=False  


import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor
from os import cpu_count
from copy import deepcopy
from random import sample
from time import perf_counter
from scipy.stats import dirichlet
from sklearn.utils import all_estimators
import warnings
from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore', category=UndefinedMetricWarning)

removed_regressors = (
    "TheilSenRegressor",
    "ARDRegression", 
    "CCA", 
    "DummyRegressor",
    "ElasticNetCV",
    "GradientBoostingRegressor",
    "HistGradientBoostingRegressor",
    "LarsCV",
    "LassoCV",
    "LassoLarsCV",
    "RidgeCV",
    "OrthogonalMatchingPursuitCV",
    "MLPRegressor",
    "SGDRegressor",
    "IsotonicRegression", 
    "StackingRegressor",
    "MultiOutputRegressor", 
    "MultiTaskElasticNet", 
    "MultiTaskElasticNetCV", 
    "MultiTaskLasso", 
    "MultiTaskLassoCV", 
    "PLSCanonical", 
    "PLSRegression", 
    "RadiusNeighborsRegressor", 
    "RegressorChain", 
    "VotingRegressor", 
)

REGRESSORS = [
    est[1]
    for est in all_estimators()
    if (issubclass(est[1], RegressorMixin) and (est[0] not in removed_regressors))
]


cdef update_posterior_probabilities(models, prior_probabilities_all, penalty_factor=0.6, num_samples=1_000_000):
    models_sorted = sorted(models, key=lambda x: x['loss'])
    for rank, model in enumerate(models_sorted, start=1):
        model['loss'] = rank
    observed_errors = np.array([model['loss'] for model in models_sorted])
    trained_model_instances = [model['model'] for model in models_sorted]
    prior_probabilities = np.array([
        prior_probabilities_all[type(model_instance)]
        for model_instance in trained_model_instances
    ])
    alpha = np.ones(len(trained_model_instances))
    samples = dirichlet.rvs(alpha, size=num_samples)
    weights = np.exp(-samples @ observed_errors)
    normalized_weights = weights / np.sum(weights)
    updated_posterior_probabilities_trained = prior_probabilities * np.dot(normalized_weights, samples)
    updated_posterior_probabilities_all = deepcopy(prior_probabilities_all)
    for i, model_instance in enumerate(trained_model_instances):
        updated_posterior_probabilities_all[type(model_instance)] = updated_posterior_probabilities_trained[i]
    untrained_model_instances = [
        model_instance
        for model_instance in prior_probabilities_all.keys()
        if not any(isinstance(model_instance, type(trained_instance)) for trained_instance in trained_model_instances)
    ]
    for model_instance in untrained_model_instances:
        updated_posterior_probabilities_all[model_instance] *= penalty_factor
    total_probability = sum(updated_posterior_probabilities_all.values())
    return {k: v / total_probability for k, v in updated_posterior_probabilities_all.items()}

class MSBoostRegressor(BaseEstimator, RegressorMixin):
    """A Multi-Stage Boosting Regressor using an ensemble of models."""

    def __init__(
        self,
        early_stopping=False,
        early_stopping_min_delta=0.001,
        early_stopping_patience=10,
        custom_models=None,
        learning_rate=0.01,
        n_estimators=100,
        custom_loss_metrics=None,
        bayes=False,
        n_models=5,
        n_iter_models=5,
        n_random_models=12,
        bayes_penalty_factor=0.5,
        bayes_random_factor=0.2,
        return_vals=True,
        return_best=True
    ):
        self.min_models_ = []  # Store the ensemble of models
        self.early_stopping = early_stopping
        self.early_stopping_min_delta = early_stopping_min_delta
        self.early_stopping_patience = early_stopping_patience
        self.custom_models = custom_models
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.custom_loss_metrics = custom_loss_metrics
        self.bayes = bayes
        self.n_models = n_models
        self.n_iter_models = n_iter_models
        self.n_random_models = n_random_models
        self.bayes_penalty_factor = bayes_penalty_factor
        self.bayes_random_factor = bayes_random_factor
        self.return_vals = return_vals
        self.return_best = return_best

    def _metrics(self, vt, vp, model, time=None):
        """Calculate loss metrics for a model."""
        if self.custom_loss_metrics:
            return {'model': model, 'time': time, 'loss': self.custom_loss_metrics(vt, vp)}
        return {"model": model, "time": time, "loss": mean_squared_error(vt, vp)}

    def _create_model(self, X, y, model_name, time_it=False):
        """Fit a model instance."""
        model = model_name()
        if time_it:
            begin = perf_counter()
            model.fit(X, y)
            end = perf_counter()
            return (model, end - begin)
        return (model.fit(X, y), None)

    def _get_metrics(self, args):
        """Combine model creation and metrics calculation for threading."""
        model_name, X, y = args
        try:
            Xt, Xv, yt, yv = train_test_split(X, y, random_state=42)
            results = self._create_model(Xt, yt, model_name, time_it=False)
            model, time = results[0], results[1]
            return self._metrics(yv, model.predict(Xv), model, time)
        except Exception:
            return None

    def _get_results(self, X, y):
        """Use ThreadPoolExecutor to evaluate all models."""
        args = [(model_name, X, y) for model_name in self._models]
        num_workers = min(cpu_count(), len(self._models))
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(self._get_metrics, args))
        return [r for r in results if r is not None]

    def fit(
        self, X, y,
        early_stopping=False,
        early_stopping_min_delta=0.001,
        early_stopping_patience=10,
        custom_models=None,
        learning_rate=0.01,
        n_estimators=100,
        custom_loss_metrics=None,
        bayes=False,
        n_models=5,
        n_iter_models=5,
        n_random_models=12,
        bayes_penalty_factor=0.5,
        bayes_random_factor=0.2,
        return_vals=True,
        return_best=True,
    ):
        X, y = check_X_y(X, y)
        self.n_features_in_ = X.shape[1]
        self._y_mean = np.mean(y)  # Initial prediction as mean
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.custom_loss_metrics = custom_loss_metrics
        self.return_best = return_best

        # Updated model list (single comprehensive set)
        if custom_models:
            self._models = custom_models
            self._models_lst = custom_models
        else:
            self._models_lst = REGRESSORS
            self._models = deepcopy(self._models_lst)

        self.min_models_ = []
        p = np.full(len(y), self._y_mean, dtype=np.float64)  # Cumulative prediction
        errors = []

        if bayes:
            self.prior_proba = dict(zip(self._models_lst, [1/len(self._models_lst)]*len(self._models_lst)))

        for i in range(self.n_estimators):
            residual = y - p  # Compute residual
            results = self._get_results(X, residual)

            if n_random_models > 0:
                self._models = tuple(sample(self._models_lst, min(n_random_models, len(self._models_lst))))
            elif bayes:
                self.prior_proba = update_posterior_probabilities(
                    models=results, prior_probabilities_all=self.prior_proba, penalty_factor=bayes_penalty_factor)
                sorted_models = sorted(self.prior_proba.items(), key=lambda x: x[1], reverse=True)
                top_n = int(n_models * (1 - bayes_random_factor))
                if n_iter_models > 0:
                    n_iter_models -= 1
                else:
                    self._models = [model for model, _ in sorted_models[:top_n]] + sample(
                        [model for model, _ in sorted_models[top_n:]], int(n_models * bayes_random_factor))

            try:
                min_loss = min(results, key=lambda x: x.get("loss", float('inf')))["loss"]
                min_model = [r['model'] for r in results if r['loss'] == min_loss][0]
            except ValueError:
                break  # No valid results, stop fitting

            p += self.learning_rate * min_model.predict(X)
            self.min_models_.append(min_model)
            error = mean_squared_error(y, p)
            errors.append(error)

            if error < 1e-8 and return_best:
                break

        if errors and return_best:
            min_error_idx = np.argmin(errors)
            self.min_models_ = self.min_models_[:min_error_idx + 1]
            errors = errors[:min_error_idx + 1]

        if return_vals:
            self.errors = errors
            self.ensemble = self.min_models_

        return self

    def predict(self, X):
        check_is_fitted(self, 'min_models_')
        X = check_array(X)
        p = np.full(X.shape[0], self._y_mean, dtype=np.float64)
        for model in self.min_models_:
            p += self.learning_rate * model.predict(X)
        return p

class MSBoostClassifier(BaseEstimator, ClassifierMixin):
    """A Multi-Stage Boosting Classifier for binary classification."""

    def __init__(
        self,
        early_stopping=False,
        early_stopping_min_delta=0.001,
        early_stopping_patience=10,
        custom_models=None,
        learning_rate=0.01,
        n_estimators=100,
        custom_loss_metrics=None,
        bayes=False,
        n_models=5,
        n_iter_models=5,
        n_random_models=12,
        bayes_penalty_factor=0.5,
        bayes_random_factor=0.2,
        return_vals=True,
        return_best=True
    ):
        self.min_models_ = []
        self.early_stopping = early_stopping
        self.early_stopping_min_delta = early_stopping_min_delta
        self.early_stopping_patience = early_stopping_patience
        self.custom_models = custom_models
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.custom_loss_metrics = custom_loss_metrics
        self.bayes = bayes
        self.n_models = n_models
        self.n_iter_models = n_iter_models
        self.n_random_models = n_random_models
        self.bayes_penalty_factor = bayes_penalty_factor
        self.bayes_random_factor = bayes_random_factor
        self.return_vals = return_vals
        self.return_best = return_best

    def _metrics(self, vt, vp, model, time=None):
        if self.custom_loss_metrics:
            return {'model': model, 'time': time, 'loss': self.custom_loss_metrics(vt, vp)}
        return {"model": model, "time": time, "loss": mean_squared_error(vt, vp)}

    def _create_model(self, X, y, model_name, time_it=False):
        model = model_name()
        if time_it:
            begin = perf_counter()
            model.fit(X, y)
            end = perf_counter()
            return (model, end - begin)
        return (model.fit(X, y), None)

    def _get_metrics(self, args):
        model_name, X, y = args
        try:
            Xt, Xv, yt, yv = train_test_split(X, y, random_state=42)
            results = self._create_model(Xt, yt, model_name, time_it=False)
            model, time = results[0], results[1]
            return self._metrics(yv, model.predict(Xv), model, time)
        except Exception:
            return None

    def _get_results(self, X, y):
        args = [(model_name, X, y) for model_name in self._models]
        num_workers = min(cpu_count(), len(self._models))
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(self._get_metrics, args))
        return [r for r in results if r is not None]

    def fit(
        self, X, y,
        early_stopping=False,
        early_stopping_min_delta=0.001,
        early_stopping_patience=10,
        custom_models=None,
        learning_rate=0.01,
        n_estimators=100,
        custom_loss_metrics=None,
        bayes=False,
        n_models=5,
        n_iter_models=5,
        n_random_models=12,
        bayes_penalty_factor=0.5,
        bayes_random_factor=0.2,
        return_vals=True,
        return_best=True,
    ):
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("MSBoostClassifier currently supports only binary classification.")
        self.n_features_in_ = X.shape[1]
        self._y_mean = np.log(np.mean(y) / (1 - np.mean(y) + 1e-15))
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.custom_loss_metrics = custom_loss_metrics
        self.return_best = return_best

        if custom_models:
            self._models = custom_models
            self._models_lst = custom_models
        else:
            self._models_lst = REGRESSORS
            self._models = deepcopy(self._models_lst)

        self.min_models_ = []
        p = np.full(len(y), self._y_mean, dtype=np.float64)
        errors = []

        if bayes:
            self.prior_proba = dict(zip(self._models_lst, [1/len(self._models_lst)]*len(self._models_lst)))

        for i in range(self.n_estimators):
            p_prob = 1 / (1 + np.exp(-p))
            residual = y - p_prob
            results = self._get_results(X, residual)

            if n_random_models > 0:
                self._models = tuple(sample(self._models_lst, min(n_random_models, len(self._models_lst))))
            elif bayes:
                self.prior_proba = update_posterior_probabilities(
                    models=results, prior_probabilities_all=self.prior_proba, penalty_factor=bayes_penalty_factor)
                sorted_models = sorted(self.prior_proba.items(), key=lambda x: x[1], reverse=True)
                top_n = int(n_models * (1 - bayes_random_factor))
                if n_iter_models > 0:
                    n_iter_models -= 1
                else:
                    self._models = [model for model, _ in sorted_models[:top_n]] + sample(
                        [model for model, _ in sorted_models[top_n:]], int(n_models * bayes_random_factor))

            try:
                min_loss = min(results, key=lambda x: x.get("loss", float('inf')))["loss"]
                min_model = [r['model'] for r in results if r['loss'] == min_loss][0]
            except ValueError:
                break

            p += self.learning_rate * min_model.predict(X)
            self.min_models_.append(min_model)
            p_prob = 1 / (1 + np.exp(-p))
            error = -np.mean(y * np.log(p_prob + 1e-15) + (1 - y) * np.log(1 - p_prob + 1e-15))
            errors.append(error)

            if error < 1e-8 and return_best:
                break

        if errors and return_best:
            min_error_idx = np.argmin(errors)
            self.min_models_ = self.min_models_[:min_error_idx + 1]
            errors = errors[:min_error_idx + 1]

        if return_vals:
            self.errors = errors
            self.ensemble = self.min_models_

        return self

    def predict_proba(self, X):
        check_is_fitted(self, 'min_models_')
        X = check_array(X)
        p = np.full(X.shape[0], self._y_mean, dtype=np.float64)
        for model in self.min_models_:
            p += self.learning_rate * model.predict(X)
        p_prob = 1 / (1 + np.exp(-p))
        return np.vstack([1 - p_prob, p_prob]).T

    def predict(self, X):
        check_is_fitted(self, 'min_models_')
        proba = self.predict_proba(X)
        return (proba[:, 1] > 0.5).astype(int)
