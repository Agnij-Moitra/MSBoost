import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from multiprocessing import Pool, cpu_count
from copy import deepcopy
from random import sample
from time import perf_counter
from scipy.stats import dirichlet

# Updated model imports
from sklearn.linear_model import (
    LinearRegression, Ridge, RidgeCV, Lasso, LassoCV, ElasticNet, ElasticNetCV,
    Lars, LassoLars, OrthogonalMatchingPursuit, SGDRegressor, PassiveAggressiveRegressor,
    RANSACRegressor, TheilSenRegressor, HuberRegressor,
    BayesianRidge, ARDRegression, QuantileRegressor, PoissonRegressor, GammaRegressor,
    TweedieRegressor
)
from sklearn.svm import SVR, NuSVR, LinearSVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor, ExtraTreesRegressor, BaggingRegressor,
)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.cross_decomposition import PLSRegression, CCA
# from sklearn.isotonic import IsotonicRegression

# Bayesian posterior update function (unchanged)
def update_posterior_probabilities(models, prior_probabilities_all, penalty_factor=0.6, num_samples=1_000_000):
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
        """Initialize the MSBoostRegressor.

        Parameters:
        -----------
        early_stopping : bool, default=False
            Whether to stop training early if performance stops improving.
        early_stopping_min_delta : float, default=0.001
            Minimum change in loss to qualify as an improvement for early stopping.
        early_stopping_patience : int, default=10
            Number of iterations with no improvement after which training stops.
        custom_models : list, default=None
            Custom list of model classes to use instead of the default set.
        learning_rate : float, default=0.01
            Rate at which model predictions contribute to the ensemble.
        n_estimators : int, default=100
            Number of boosting iterations or models to fit.
        custom_loss_metrics : callable, default=None
            Custom loss function taking true and predicted values as arguments.
        bayes : bool, default=False
            Whether to use Bayesian model selection to update model probabilities.
        n_models : int, default=5
            Number of top models to select in Bayesian mode.
        n_iter_models : int, default=5
            Number of iterations before finalizing Bayesian model selection.
        n_random_models : int, default=12
            Number of models to randomly sample each iteration (if > 0).
        bayes_penalty_factor : float, default=0.5
            Penalty factor for untrained models in Bayesian updates.
        bayes_random_factor : float, default=0.2
            Fraction of models to randomly select in Bayesian mode.
        return_vals : bool, default=True
            Whether to store and return errors and ensemble details.
        return_best : bool, default=True
            Whether to return the best ensemble based on minimum error.
        """

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
        """Combine model creation and metrics calculation for multiprocessing."""
        model_name, X, y = args
        try:
            Xt, Xv, yt, yv = train_test_split(X, y, random_state=42)
            results = self._create_model(Xt, yt, model_name, time_it=False)
            model, time = results[0], results[1]
            return self._metrics(yv, model.predict(Xv), model, time)
        except Exception:
            return None

    def _get_results(self, X, y):
        """Use multiprocessing to evaluate all models."""
        # Prepare arguments for each model
        args = [(model_name, X, y) for model_name in self._models]
        # Use number of CPU cores as the default number of processes, capped by number of models
        num_processes = min(cpu_count(), len(self._models))
        with Pool(processes=num_processes) as pool:
            results = pool.map(self._get_metrics, args)
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
        """Fit the MSBoostRegressor model."""
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
            self._models_lst = (
                LinearRegression, Ridge, RidgeCV, Lasso, LassoCV, ElasticNet, ElasticNetCV,
                Lars, LassoLars, OrthogonalMatchingPursuit, SGDRegressor, PassiveAggressiveRegressor,
                RANSACRegressor, TheilSenRegressor, HuberRegressor,
                BayesianRidge, ARDRegression, QuantileRegressor, PoissonRegressor, GammaRegressor,
                TweedieRegressor, SVR, NuSVR, LinearSVR, KNeighborsRegressor,
                DecisionTreeRegressor, ExtraTreeRegressor, RandomForestRegressor, ExtraTreesRegressor,
                BaggingRegressor, GaussianProcessRegressor, KernelRidge, PLSRegression, CCA,
            )
            self._models = deepcopy(self._models_lst)

        # Initialize ensemble and predictions
        self.min_models_ = []
        p = np.full(len(y), self._y_mean, dtype=np.float64)  # Cumulative prediction
        errors = []

        if bayes:
            self.prior_proba = dict(zip(self._models_lst, [1/len(self._models_lst)]*len(self._models_lst)))

        for i in range(self.n_estimators):
            residual = y - p  # Compute residual
            results = self._get_results(X, residual)

            # Model selection logic
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

            if error < 1e-6 and return_best:
                break

        # Select best ensemble based on minimum error
        if errors and return_best:
            min_error_idx = np.argmin(errors)
            self.min_models_ = self.min_models_[:min_error_idx + 1]
            errors = errors[:min_error_idx + 1]

        if return_vals:
            self.errors_ = errors
            self.ensemble_ = self.min_models_

        return self

    def predict(self, X):
        """Predict using the fitted ensemble."""
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
        """Initialize the MSBoostClassifier for binary classification.

        Parameters:
        -----------
        early_stopping : bool, default=False
            Whether to stop training early if performance stops improving.
        early_stopping_min_delta : float, default=0.001
            Minimum change in loss to qualify as an improvement for early stopping.
        early_stopping_patience : int, default=10
            Number of iterations with no improvement after which training stops.
        custom_models : list, default=None
            Custom list of model classes to use instead of the default set.
        learning_rate : float, default=0.01
            Rate at which model predictions contribute to the ensemble.
        n_estimators : int, default=100
            Number of boosting iterations or models to fit.
        custom_loss_metrics : callable, default=None
            Custom loss function taking true and predicted values as arguments.
        bayes : bool, default=False
            Whether to use Bayesian model selection to update model probabilities.
        n_models : int, default=5
            Number of top models to select in Bayesian mode.
        n_iter_models : int, default=5
            Number of iterations before finalizing Bayesian model selection.
        n_random_models : int, default=12
            Number of models to randomly sample each iteration (if > 0).
        bayes_penalty_factor : float, default=0.5
            Penalty factor for untrained models in Bayesian updates.
        bayes_random_factor : float, default=0.2
            Fraction of models to randomly select in Bayesian mode.
        return_vals : bool, default=True
            Whether to store and return errors and ensemble details.
        return_best : bool, default=True
            Whether to return the best ensemble based on minimum error.
        """
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
        """Combine model creation and metrics calculation for multiprocessing."""
        model_name, X, y = args
        try:
            Xt, Xv, yt, yv = train_test_split(X, y, random_state=42)
            results = self._create_model(Xt, yt, model_name, time_it=False)
            model, time = results[0], results[1]
            return self._metrics(yv, model.predict(Xv), model, time)
        except Exception:
            return None

    def _get_results(self, X, y):
        """Use multiprocessing to evaluate all models."""
        # Prepare arguments for each model
        args = [(model_name, X, y) for model_name in self._models]
        # Use number of CPU cores as the default number of processes, capped by number of models
        num_processes = min(cpu_count(), len(self._models))
        with Pool(processes=num_processes) as pool:
            results = pool.map(self._get_metrics, args)
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
        """Fit the MSBoostClassifier model (binary classification)."""
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("MSBoostClassifier currently supports only binary classification.")
        self.n_features_in_ = X.shape[1]
        self._y_mean = np.log(np.mean(y) / (1 - np.mean(y) + 1e-15))  # Initial log-odds
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.custom_loss_metrics = custom_loss_metrics
        self.return_best = return_best

        # Updated model list (single comprehensive set)
        if custom_models:
            self._models = custom_models
            self._models_lst = custom_models
        else:
            self._models_lst = (
                LinearRegression, Ridge, RidgeCV, Lasso, LassoCV, ElasticNet, ElasticNetCV,
                Lars, LassoLars, OrthogonalMatchingPursuit, SGDRegressor, PassiveAggressiveRegressor,
                RANSACRegressor, TheilSenRegressor, HuberRegressor,
                BayesianRidge, ARDRegression, QuantileRegressor, PoissonRegressor, GammaRegressor,
                TweedieRegressor, SVR, NuSVR, LinearSVR, KNeighborsRegressor,
                DecisionTreeRegressor, ExtraTreeRegressor, RandomForestRegressor, ExtraTreesRegressor,
                BaggingRegressor, GaussianProcessRegressor, KernelRidge, PLSRegression, CCA,
            )
            self._models = deepcopy(self._models_lst)

        # Initialize ensemble and predictions
        self.min_models_ = []
        p = np.full(len(y), self._y_mean, dtype=np.float64)  # Cumulative prediction in log-odds
        errors = []

        if bayes:
            self.prior_proba = dict(zip(self._models_lst, [1/len(self._models_lst)]*len(self._models_lst)))

        for i in range(self.n_estimators):
            p_prob = 1 / (1 + np.exp(-p))  # Sigmoid to get probabilities
            residual = y - p_prob          # Gradient of logistic loss
            results = self._get_results(X, residual)

            # Model selection logic
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

            # Update prediction
            p += self.learning_rate * min_model.predict(X)
            self.min_models_.append(min_model)
            p_prob = 1 / (1 + np.exp(-p))
            error = -np.mean(y * np.log(p_prob + 1e-15) + (1 - y) * np.log(1 - p_prob + 1e-15))  # Log loss
            errors.append(error)

            if error < 1e-6 and return_best:
                break

        # Select best ensemble based on minimum error
        if errors and return_best:
            min_error_idx = np.argmin(errors)
            self.min_models_ = self.min_models_[:min_error_idx + 1]
            errors = errors[:min_error_idx + 1]

        if return_vals:
            self.errors_ = errors
            self.ensemble_ = self.min_models_

        return self

    def predict_proba(self, X):
        """Predict class probabilities."""
        check_is_fitted(self, 'min_models_')
        X = check_array(X)
        p = np.full(X.shape[0], self._y_mean, dtype=np.float64)
        for model in self.min_models_:
            p += self.learning_rate * model.predict(X)
        p_prob = 1 / (1 + np.exp(-p))
        return np.vstack([1 - p_prob, p_prob]).T  # [P(y=0), P(y=1)]

    def predict(self, X):
        """Predict class labels."""
        check_is_fitted(self, 'min_models_')
        proba = self.predict_proba(X)
        return (proba[:, 1] > 0.5).astype(int)  # Threshold at 0.5

# # Example usage
if __name__ == "__main__":
    from sklearn.datasets import make_regression, make_classification

    # Regression example
    X_reg, y_reg = make_regression(n_samples=10, n_features=10, noise=0.1, random_state=42)
    reg = MSBoostRegressor()
    reg.fit(X_reg, y_reg)
    y_pred_reg = reg.predict(X_reg)
    print("Regression MSE:", mean_squared_error(y_reg, y_pred_reg))

    # Classification example
    X_clf, y_clf = make_classification(n_samples=10, n_features=10, n_classes=2, random_state=42)
    clf = MSBoostClassifier()
    clf.fit(X_clf, y_clf)
    y_pred_clf = clf.predict(X_clf)
    proba = clf.predict_proba(X_clf)
    print("Classification Accuracy:", (y_pred_clf == y_clf).mean())
    print("Sample probabilities:", proba[:5])