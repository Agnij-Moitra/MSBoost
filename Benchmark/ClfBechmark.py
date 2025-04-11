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
from tqdm import tqdm
import pandas as pd
import os
import sys
import contextlib
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    balanced_accuracy_score, matthews_corrcoef, jaccard_score
)

from sklearn.ensemble import GradientBoostingClassifier
from MSBoost import MSBoostClassifier, SnapBoostClassifier


# ---------------------
# UTILITY FUNCTIONS
# ---------------------

@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

def pre_process_y(y):
    return LabelEncoder().fit_transform(y)

def get_card_split(df, cols, n=11):
    cond = df[cols].nunique() > n
    card_high = cols[cond]
    card_low = cols[~cond]
    return card_low, card_high

def append_row(df, data):
    df.loc[len(df)] = data


# ---------------------
# CLASSIFICATION EVALUATION
# ---------------------

def evaluate_binary_split(model_cls, X, y):
    """
    Evaluate a classifier using 70-30 train-test split.

    Returns:
        Tuple of 7 metrics:
        (accuracy, f1, precision, recall, balanced_acc, mcc, jaccard)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=7
    )
    model = model_cls(n_estimators=100)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return (
        accuracy_score(y_test, y_pred),
        f1_score(y_test, y_pred),
        precision_score(y_test, y_pred),
        recall_score(y_test, y_pred),
        balanced_accuracy_score(y_test, y_pred),
        matthews_corrcoef(y_test, y_pred),
        jaccard_score(y_test, y_pred),
    )


# ---------------------
# PIPELINE SETUP
# ---------------------

numeric_transformer = Pipeline(
    steps=[("imputer", KNNImputer(n_neighbors=15, weights="distance")), ("scaler", StandardScaler())]
)

categorical_transformer_low = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encoding", OneHotEncoder(handle_unknown="ignore")),
    ]
)

categorical_transformer_high = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encoding", OrdinalEncoder()),
    ]
)

# ---------------------
# DATASET CONFIG
# ---------------------

data_id = set([
    151, 293, 722, 821, 1120, 1461, 4134, 41150, 42477, 42769,
    1044, 4541, 41168, 45026, 44089, 45028, 1596, 41147,
    42192, 42803, 40536, 1489, 4532, 40981, 25, 41159,
    1169, 41143
])



num_samples = 1000
cols = [
    "Dataset", "OpenML ID", "Model",
    "Accuracy", "F1", "Precision", "Recall",
    "Balanced Acc", "MCC", "Jaccard"
]

benchmark = pd.DataFrame(columns=cols)
invalid = []

def clf_bench():
    for i, ID in enumerate(tqdm(sorted(data_id), desc="Benchmarking", unit="dataset")):
        try:
            with suppress_stdout():
                data = fetch_openml(data_id=ID, as_frame=True)
            name = data["details"]["name"]
            df = data["frame"]
            target = data["target_names"]

            if len(df) > num_samples:
                df = df.sample(num_samples, random_state=7)

            X, y = df.drop(target, axis=1), df[target]

            numeric_features = X.select_dtypes(include='number').columns
            categorical_features = X.select_dtypes(include=["object", "category"]).columns

            cat_low, cat_high = get_card_split(X, categorical_features)

            preprocessor = ColumnTransformer(
                transformers=[
                    ("numeric", numeric_transformer, numeric_features),
                    ("categorical_low", categorical_transformer_low, cat_low),
                    ("categorical_high", categorical_transformer_high, cat_high),
                ]
            )

            X = preprocessor.fit_transform(X)
            y = pre_process_y(y[y.columns[0]])

            for model_name, model_cls in [
                ("GB", GradientBoostingClassifier),
                ("MSBoost", MSBoostClassifier),
                ("SnapBoost", SnapBoostClassifier)
            ]:
                try:
                    with suppress_stdout():
                        metrics = evaluate_binary_split(model_cls, X, y)
                    append_row(benchmark, [name, ID, model_name] + list(metrics))
                except Exception as e:
                    invalid.append((ID, model_name, repr(e)))

            benchmark.to_csv("OpenML_binary_metrics.csv", index=False)

        except Exception as e:
            invalid.append((ID, "load_failure", repr(e)))
        finally:
            print(invalid)

if __name__ == "__main__":
    clf_bench()