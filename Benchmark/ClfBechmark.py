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
from MSBoost import MSBoostClassifier, SnapBoostClassifier  # Assumed available in your environment
from concurrent.futures import ProcessPoolExecutor
import warnings
warnings.filterwarnings("ignore")

# Utility Functions
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

# Classification Evaluation
def evaluate_binary_split(model_cls, X, y):
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=7
        )
        model = model_cls(n_estimators=100)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return (
            accuracy_score(y_test, y_pred),
            f1_score(y_test, y_pred, average='binary', zero_division=0),
            precision_score(y_test, y_pred, average='binary', zero_division=0),
            recall_score(y_test, y_pred, average='binary', zero_division=0),
            balanced_accuracy_score(y_test, y_pred),
            matthews_corrcoef(y_test, y_pred),
            jaccard_score(y_test, y_pred, average='binary', zero_division=0),
        )
    except Exception as e:
        return tuple([float('nan')] * 7)  # Return NaN for all metrics if evaluation fails

# Pipeline Setup
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

# Dataset Config
data_id = [
    151, 293, """722, 821, 1120, 1461, 4134, 41150, 42477, 42769,
    1044, 4541, 41168, 45026, 44089, 45028, 1596, 41147,
    42192, 42803, 40536, 1489, 4532, 40981, 25, 41159,
    1169, 41143"""
]
num_samples = 1000
cols = [
    "Dataset", "OpenML ID", "Model",
    "Accuracy", "F1", "Precision", "Recall",
    "Balanced Acc", "MCC", "Jaccard"
]
benchmark = pd.DataFrame(columns=cols)
invalid = []

# Process Single Dataset
def process_dataset(ID):
    local_benchmark = pd.DataFrame(columns=cols)
    local_invalid = []
    try:
        with suppress_stdout():
            data = fetch_openml(data_id=ID, as_frame=True)  # Pass ID as int
        name = data["details"]["name"]
        df = data["frame"]
        target = data["target_names"]

        # Check if target is binary
        y = df[target]
        if y.nunique().iloc[0] != 2:
            local_invalid.append((ID, "load_failure", "Non-binary target"))
            return local_benchmark, local_invalid

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
                append_row(local_benchmark, [name, ID, model_name] + list(metrics))
            except Exception as e:
                local_invalid.append((ID, model_name, repr(e)))

    except Exception as e:
        local_invalid.append((ID, "load_failure", repr(e)))

    return local_benchmark, local_invalid

# Main Function
def clf_bench():
    global benchmark, invalid
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(process_dataset, data_id))
    
    for local_benchmark, local_invalid in results:
        benchmark = pd.concat([benchmark, local_benchmark], ignore_index=True)
        invalid.extend(local_invalid)
    
    benchmark.to_csv("OpenML_binary_metrics.csv", index=False)
    print("Invalid cases:", invalid)

if __name__ == "__main__":
    clf_bench()