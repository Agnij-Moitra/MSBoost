import numpy as np
import pandas as pd
from collections import Counter
from pandas import DataFrame, concat
from concurrent.futures import ThreadPoolExecutor
# Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, f1_score
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, BaggingRegressor, ExtraTreesRegressor, RandomForestRegressor, RandomForestClassifier
from sklearn.svm import NuSVR, SVC, SVR, LinearSVR
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from lightgbm import LGBMRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression, BayesianRidge, LassoCV, RidgeCV, LarsCV, OrthogonalMatchingPursuitCV, LassoLarsCV, ElasticNet, ElasticNetCV, SGDRegressor, LassoLars, Lasso, Ridge, ARDRegression, RANSACRegressor, HuberRegressor, TheilSenRegressor, LassoLarsIC
from sklearn.kernel_ridge import KernelRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from time import perf_counter
from lazypredict.Supervised import LazyClassifier
from random import sample
from copy import deepcopy
from catboost import CatBoostRegressor

class MSBoostClassifier(BaseEstimator, ClassifierMixin):
    """A Gradient Boosting Classifier
    """

    def __init__(self, **kwargs):
        """ Initialize MSBoost Object
        """

    def _metrics(self, vt, vp, model, time=None):
        """get loss metrics of a model

        Args:
            vt (iterable): validation true values
            vp (iterable): validation pred values
            model (object): any model with fit and predict method
            time (float, optional): execution time of the model. Defaults to None.

        Returns:
            dict['model', 'time', 'loss']
        """
        if self.custom_loss_metrics:
            return {'model': model, 'time': time, 'loss': self.custom_loss_metrics(vt, vp)}
        
        # vp = 1 / (1 + np.exp(-vp))
        # vp = np.column_stack((1 - vp, vp))
        
        return {"model": model, "time": time, "loss": mean_squared_error(vt, vp)}

    def _create_model(self, X, y, model_name, time_it: bool = False):
        """fit a model instance

        Args:
            X (iterable)
            y (iterable)
            model_name (object): any model object with fit and predict methods
            time_it (bool, optional): measure execution time. Defaults to False.

        Returns:
            tuple(model, time=None)
        """
        model = model_name()
        if time_it:
            begin = perf_counter()
            model.fit(X, y)
            end = perf_counter()
            return (model, end - begin)
        return (model.fit(X, y), None)

    def _get_metrics(self, model_name):
        """a helper fuction, combines self._create_model and self._metrics

        Args:
            model_name (object): any model with fit and predict methods

        Returns:
            self._metrics
        """
        try:
            Xt, Xv, yt, yv = train_test_split(self._X, self._y, train_size=0.9)
            results = self._create_model(Xt, yt, model_name, time_it=False)
            model, time = results[0], results[1]
            return self._metrics(yv,
                                 model.predict(Xv), model, time)
        except Exception:
            return None

    def _get_results(self, X, y) -> list:
        """Use multi-threading to return all results

        Args:
            X (iterable)
            y (iterable)

        Returns:
            list[dict['model', 'time', 'loss']]
        """
        results = []
        # self._X = self._minimax.fit_transform(self._robust.fit_transform(
        #         KNNImputer(weights='distance').fit_transform(X)))
        self._X = X
        self._y = y
        with ThreadPoolExecutor(max_workers=len(self._models)) as executor:
            res = executor.map(self._get_metrics, self._models)
            results = [i for i in res if i]
        return results

    def fit(
        self, X, y,
        early_stopping: bool = False,
        early_stopping_min_delta: float = 0.001,
        early_stopping_patience: int = 10,
        custom_models: list = None,
        learning_rate: float = 0.1,
        n_estimators: int = 100,
        warm_start: bool = False,
        complexity: bool = False,
        light: bool = False,
        custom_loss_metrics: object = False,
        freeze_models: bool = False,
        n_models: int = 5,
        n_iter_models: int = 5,
        n_warm: int = None,
        n_random_models: int = 6,
        return_vals: bool = True,
    ):
        """fit MSBoost model

        Args:
            X (iterable)
            y (iterbale)
            early_stopping (bool, optional): Defaults to False.
            early_stopping_min_delta (float, optional): Defaults to 0.001.
            early_stopping_patience (int, optional): Defaults to 10.
            custom_models (tuple, optional): tuple of custom models with fit and predict methods. Defaults to None.
            learning_rate (float, optional): Defaults to 0.05.
            n_estimators (int, optional): Defaults to 100.
            warm_start (bool, optional): Defaults to False.
            complexity (bool, optional): trains more models but has greater time complexity. Defaults to False.
            light (bool, optional): trains less models. Defaults to True.
            custom_loss_metrics (object, optional): _description_. Defaults to False.
            freeze_models (bool, optional): test only a selected models. Defaults to False.
            n_models (int, optional): Applicable for freeze_models, number of models to train. Defaults to 5.
            n_iter_models (int, optional): Applicable for freeze_models, number of iterations before finalizing the models. Defaults to 5.
            n_warm (int, optional): Applicable for warm start, number of iterarions to store. Defaults to None.
            n_random_models (int, optional): train on a random number of models. Defaults to 0.
            return_vals (bool, optional): returns analytics. Defaults to True.

        Returns:
            tuple[final ensemble sequence, mean absolute error of each layer, residual value of each layer],
            None
        """
        X, y = check_X_y(X, y)
        self.classes_ = np.array(set(y))
        self.y_max = max(y)
        # self.n_classes_ = len(self.classes_)
        self.len_X = X.shape[0]
        self.n_features_in_ = X.shape[1]
        if custom_models:
            self._models = custom_models
        self.custom_loss_metrics = custom_loss_metrics
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.early_stopping = early_stopping
        self.early_stopping_min_delta = early_stopping_min_delta
        self.early_stopping_patience = early_stopping_patience
        if custom_models:

            self._models_lst = custom_models
        else:
            if complexity:
                self._models_lst = (DecisionTreeRegressor, LinearRegression, BayesianRidge, KNeighborsRegressor, HistGradientBoostingRegressor,
                                    ElasticNet, LassoLars, Lasso, GradientBoostingRegressor, ExtraTreesRegressor, SVC,
                                    BaggingRegressor, NuSVR, XGBRegressor, SGDRegressor, KernelRidge, MLPRegressor, LGBMRegressor,
                                    Ridge, ARDRegression, RANSACRegressor, HuberRegressor, TheilSenRegressor, LassoLarsIC)
            elif light:
                self._models_lst = (LGBMRegressor, ExtraTreesRegressor, CatBoostRegressor, KNeighborsRegressor, LassoCV, ElasticNetCV, RidgeCV, DecisionTreeRegressor,
                                     BaggingRegressor, LassoLarsIC, BayesianRidge)
            else:
                self._models_lst = (DecisionTreeRegressor, LinearRegression, BayesianRidge, KNeighborsRegressor, LGBMRegressor,
                                    ElasticNet, LassoLars, Lasso, SGDRegressor, BaggingRegressor, ExtraTreesRegressor,
                                    Ridge, ARDRegression, RANSACRegressor, LassoLarsIC)
            self._models = deepcopy(self._models_lst)
        self.freeze_models = freeze_models
        if self.freeze_models:
            self.n_models = n_models
            self.n_iter_models = n_iter_models
        self._y_mean = 0.5
        # base model: mean
        # computer residuals: y - y hat
        # for n_estimators: a) y = prev residuals && residuals * learning rate
        # add early stopping
        # restore best weights
        # ada boost and adaptive scaling for learning rates
        self._ensemble = []
        preds = DataFrame(
            data={'yt': y, 'p0': np.full((len(y)), self._y_mean)})
        residuals = DataFrame(
            data={'r0': y - self._y_mean})
        errors = []
        if not early_stopping:
            if warm_start:
                for i in range(1, self.n_estimators + 1):
                    y = residuals[f'r{i - 1}']
                    results = self._get_results(X, y)
                    min_loss = min(results, key=lambda x: x.get(
                        "loss", float('inf')))["loss"]  # https://stackoverflow.com/a/19619294
                    min_model = [i['model']
                                 for i in results if min_loss >= i['loss']][0]
                    preds[f'p{i}'] = residuals.sum(axis=1) + min_model.predict(
                        X) * self.learning_rate
                    residuals[f'r{i}'] = preds['yt'] - preds[f'p{i}']
                    if i % n_warm == 0:
                        X[f"r{i}"] = residuals[f'r{i}'].copy()
                    try:
                        errors.append(mean_squared_error(
                            preds['yt'], preds[f'p{i}']))
                    except Exception:
                        df = concat(
                            [preds['yt'], preds[f'p{i - 1}']], axis=1).dropna()
                        errors.append(mean_squared_error(
                            df['yt'], df[f"p{i - 1}"]))
                    self._ensemble.append(min_model)
            else:
                freeze_models_lst = []
                for i in range(1, self.n_estimators + 1):
                    y = residuals[f'r{i - 1}']
                    results = self._get_results(X, y)
                    # return results
                    if n_random_models > 0:
                        self._models = tuple(
                            sample(self._models_lst, n_random_models))
                    elif self.freeze_models:
                        if self.n_iter_models > -1:
                            freeze_models_lst.append([i.get("model") for i in sorted(results, key=lambda x: x.get(
                                "loss", float('inf')))][:n_models])
                            self.n_iter_models -= 1
                        else:
                            model_lst = sorted(dict(Counter(i for sub in freeze_models_lst for i in set(
                                sub))).items(), key=lambda ele: ele[1], reverse=True)
                            # return model_lst
                            self._models = tuple(type(i[0]) for i in model_lst)[
                                :n_models]
                            # return self._models
                    # try:
                    
                    min_loss = min(results, key=lambda x: x.get(
                            "loss", float('inf')))["loss"]  # https://stackoverflow.com/a/19619294
                    # except Exception:
                    #     continue
                    min_model = [i['model']
                                 for i in results if min_loss >= i['loss']][0]
                    preds[f'p{i}'] = residuals.sum(axis=1) + min_model.predict(
                        X) * self.learning_rate
                    
#                     -(y - 1 / (1 + np.exp(-f_prev)))

                    residuals[f'r{i}'] = -(preds['yt'] - 1 / (1 + np.exp(preds[f'p{i}'])))
                    errors.append(mean_squared_error(
                        preds['yt'], preds[f'p{i}']))
                    self._ensemble.append(min_model)
                    if errors[i - 1] == 0:
                        break
        else:
            return self
        min_error = min(errors)
        min_error_i = [i for i in range(
            len(errors)) if errors[i] == min_error][0]
        self._ensemble, errors = self._ensemble[:
                                                min_error_i], errors[:min_error_i]
        residuals = residuals[:len(errors)]
        if return_vals:
            self.residuls = residuals
            self.errors = errors
            self.ensemble = self._ensemble
        return self

    def predict(self, X):
        """
        Args:
            X_test (iterable)

        Returns:
            numpy.array: predictions
        """
        check_is_fitted(self)
        X = check_array(X)
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X):
        check_is_fitted(self)
        X = check_array(X)
        preds = DataFrame(
            data={'p0': np.full((len(X)), self._y_mean)})
        for i in range(len(self._ensemble)):
            preds[f"p{i+1}"] = self._ensemble[i].predict(X)
        preds_ = preds.sum(axis=1)
        # probabilities = 1 / (1 + np.exp(-predictions))
        preds_ = 1 / (1 + np.exp(-preds_))
        return np.column_stack((1 - preds_, preds_))
