from sklearn.datasets import make_friedman3, make_regression, make_friedman2, fetch_california_housing, load_diabetes, make_friedman1, make_sparse_uncorrelated, make_classification
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import KFold, cross_val_score
from sklearn import datasets
from sklearn.metrics import log_loss, f1_score
import numpy as np
import random
np.random.seed(7)
random.seed(7)

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
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, BaggingRegressor, ExtraTreesRegressor, RandomForestRegressor, GradientBoostingClassifier
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


from random import sample
from copy import deepcopy
from catboost import CatBoostRegressor

class MSBoostClassifier(BaseEstimator, ClassifierMixin):
    """A Gradient Boosting Classifier
    """

    def __init__(self, n_estimators):
        """ Initialize MSBoostClassifier Object
        """
        self.n_estimators = n_estimators

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
            Xt, Xv, yt, yv = train_test_split(self._X, self._y)
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
        learning_rate: float = 0.05,
        n_estimators: int = 100,
        warm_start: bool = False,
        complexity: bool = True,
        light: bool = False,
        custom_loss_metrics: object = False,
        freeze_models: bool = False,
        n_models: int = 5,
        n_iter_models: int = 5,
        n_warm: int = None,
        n_random_models: int = 12,
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
                self._models_lst = (DecisionTreeRegressor, LinearRegression, BayesianRidge, KNeighborsRegressor,
                                    ElasticNet, LassoLars, Lasso, ExtraTreesRegressor, SVC, CatBoostRegressor, HistGradientBoostingRegressor, 
                                    BaggingRegressor, NuSVR, SGDRegressor, KernelRidge, MLPRegressor, XGBRegressor, LGBMRegressor, GradientBoostingRegressor,
                                    Ridge, ARDRegression, RANSACRegressor, HuberRegressor, TheilSenRegressor, LassoLarsIC)
            elif light:
                self._models_lst = (LGBMRegressor, ExtraTreesRegressor,
                                     BaggingRegressor, RANSACRegressor, LassoLarsIC, BayesianRidge)
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
            data={'yt': y, 'p0': np.full((len(y)), y - self._y_mean)})
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
                    try:
                        min_loss = min(results, key=lambda x: x.get(
                            "loss", float('inf')))["loss"]  # https://stackoverflow.com/a/19619294
                    except Exception:
                        continue
                    min_model = [i['model']
                                 for i in results if min_loss >= i['loss']][0]
                    preds[f'p{i}'] = residuals.sum(axis=1) + min_model.predict(
                        X) * self.learning_rate
                    residuals[f'r{i}'] = preds['yt'] - preds[f'p{i}']
                    errors.append(mean_squared_error(
                        preds['yt'], preds[f'p{i}']))
                    self._ensemble.append(min_model)
                    if errors[i - 1] == 0:
                        break
        else:
            return self
        # min_error = min(errors)
        # min_error_i = [i for i in range(
        #     len(errors)) if errors[i] == min_error][0]
        # self._ensemble, errors = self._ensemble[:
        #                                         min_error_i], errors[:min_error_i]
        # residuals = residuals[:len(errors)]

        return self

    def predict(self, X_test):
        """
        Args:
            X_test (iterable)

        Returns:
            numpy.array: predictions
        """
        check_is_fitted(self)
        X_test = check_array(X_test)
        # X_test = self._robust.transform(self._minimax.transform(deepcopy(X_test)))
        preds = DataFrame(
            data={'p0': np.full((len(X_test)), self._y_mean)})
        for i in range(len(self._ensemble)):
            preds[f"p{i}"] = self._ensemble[i].predict(X_test)
        preds_ = preds.sum(axis=1)

        def quantize(x):
            if x < 0:
                return 0
            elif x > self.y_max:
                return self.y_max
            else:
                return round(x)
        return preds_.apply(quantize).to_numpy()
        # return self.final_estimator.predict(preds.sum(axis=1)).to_numpy()

    def predict_proba(self, X):
        #     # crude data
        #     # dont quantize
        #     # https://datascience.stackexchange.com/q/22762
        #     # https://www.youtube.com/watch?v=ZsM2z0pTbnk
        #     # https://www.youtube.com/watch?v=yJK4sYclhg8
        check_is_fitted(self)
        X = check_array(X)
        preds = DataFrame(
            data={'p0': np.full((len(X)), self._y_mean)})
        for i in range(len(self._ensemble)):
            preds[f"p{i}"] = self._ensemble[i].predict(X)
        preds_ = MinMaxScaler().fit_transform(
            preds.sum(axis=1).to_numpy().reshape(-1, 1))
        preds_ = preds_.reshape(1, -1)[0]
        proba = []
        for i in preds_:
            proba.append([1.0 - i, i])
        return np.array(proba)

def subsample(X, y, num_samples=1000, random_state=7):
    """
    Subsample the arrays X and y.

    Parameters:
    - X: Input array
    - y: Target array
    - num_samples: Number of samples to retain
    - random_state: Seed for random number generator (default is None)

    Returns:
    - Subsampled X and y arrays
    """
    if random_state is not None:
        np.random.seed(random_state)

    indices = np.random.choice(len(y), num_samples, replace=False)

    return X[indices], y[indices]

def get_card_split(df, cols, n=11):
    """
    Splits categorical columns into 2 lists based on cardinality (i.e # of unique values)
    Parameters (Source: https://github.com/shankarpandala/lazypredict/blob/dev/lazypredict/Supervised.py#L114)
    ----------
    df : Pandas DataFrame
        DataFrame from which the cardinality of the columns is calculated.
    cols : list-like
        Categorical columns to list
    n : int, optional (default=11)
        The value of 'n' will be used to split columns.
    Returns
    -------
    card_low : list-like
        Columns with cardinality < n
    card_high : list-like
        Columns with cardinality >= n
        
    """
    cond = df[cols].nunique() > n
    card_high = cols[cond]
    card_low = cols[~cond]
    return card_low, card_high

def append_row(df, data):
    """
    Append a row of data to a DataFrame.

    Parameters :
        - df (pandas.DataFrame): The DataFrame to which the row will be appended.
        - data (list): The data representing a row to be appended. Should be a list where each element corresponds to a column in the DataFrame.

    Returns:
        None
    """
    df.loc[len(df)] = data
    
import os
import contextlib
import sys

@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout
            
def score_clf(model, X, y):
    return log_loss(y, model.predict(X))

def pre_process_y(y):
    return LabelEncoder().fit_transform(y)

def cv_evaluate(model, X, y, scoring="neg_mean_squared_error", n_jobs=1):
    reg = model(n_estimators=100)
    n_folds = 5
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=7)
    with suppress_stdout():
        cv_results = cross_val_score(reg, X, y, cv=kf, scoring=scoring, n_jobs=n_jobs)
    mean_mse = np.mean(cv_results)
    std_mse = np.std(cv_results)
    return f"{mean_mse:.4f} ± {std_mse:.4f}"

# from lightgbm import LGBMClassifier
# from catboost import CatBoostClassifier

# from xgboost import XGBClassifier

data_id = [
    151,
     293,
     722,
     821,
     1120,
     1461,
     4134,
     41150,
     42477,
     42769,
     1044,
     4541,
     41168,
     45026,
     45026,
    44089,
    45028,
    151,
    1044,
    1596,
    41147,
    42192,
    42477,
    42803,
    40536, 
    1489, 
    4134, 
    4532, 
    40981, 
    25, 
    41159, 
    1169,
    41143,    
]

# adapted from LazyPredict
from sklearn.datasets import fetch_openml
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder

numeric_transformer = Pipeline(
    steps=[("imputer", KNNImputer(n_neighbors=15, weights="distance")), ("scaler", StandardScaler())]
)

categorical_transformer_low = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encoding", OneHotEncoder(handle_unknown="ignore",)),
    ]
)

categorical_transformer_high = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        # 'OrdianlEncoder' Raise a ValueError when encounters an unknown value. Check https://github.com/scikit-learn/scikit-learn/pull/13423
        ("encoding", OrdinalEncoder()),
    ]
)




import pandas as pd

num_samples = 1000
cols = ["Dataset", "OpenML ID", "GBDT", "MSBoost"]
benchmark = pd.DataFrame(columns=cols)
lasso_threshold = 0.005 # remove irrelevant features 
invalid = []

for i in range(len(data_id)):
    try:
        ID = data_id[i]
        try:
            data = fetch_openml(data_id=ID)
        except:
            continue
        name = data["details"]["name"]
        df = data["frame"]
        target = data["target_names"]
        try:
            if len(df) > num_samples:
                df = df.sample(num_samples, random_state=7)
        except:
            continue
        X, y = df.drop(target, axis=1), df[target]

        numeric_features = X.select_dtypes(include=[np.number]).columns
        categorical_features = X.select_dtypes(include=["object"]).columns

        categorical_low, categorical_high = get_card_split(
            X, categorical_features
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("numeric", numeric_transformer, numeric_features),
                ("categorical_low", categorical_transformer_low, categorical_low),
                ("categorical_high", categorical_transformer_high, categorical_high),
            ]
        )

        X = preprocessor.fit_transform(X)
        y = pre_process_y(y[y.columns[0]])
        lasso = LassoCV(cv=10, max_iter=5000)
        lasso.fit(X, y)
        coefficients = lasso.coef_
        selected_features = np.where(np.abs(coefficients) > lasso_threshold)[0]
        X = X[:, selected_features]
        gbdt = "NaN"
        msboost = "NaN"
        with suppress_stdout():
            try:
                lgb = cv_evaluate(GradientBoostingClassifier, X, y, score_clf)
            except Exception as error:
                invalid.append(i, error)
            try:
                msboost = cv_evaluate(MSBoostClassifier, X, y, score_clf)
            except Exception as error:
                invalid.append([i, error])
        append_row(benchmark, [name, ID, gbdt, msboost])
        benchmark.to_csv("OpenML.csv", index=False)
    except Exception as error:
        invalid.append([i, error])
        
benchmark.to_csv("OpenML.csv", index=False)
