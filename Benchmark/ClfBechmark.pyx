from MSBoost import MSBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import log_loss, f1_score
from sklearn.model_selection import train_test_split

def train_test_evaluate(model_cls, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=7)
    model = model_cls(n_estimators=100)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    return (
        accuracy_score(y_test, y_pred),
        log_loss(y_test, y_prob),
        f1_score(y_test, y_pred, average="weighted"),
        precision_score(y_test, y_pred, average="weighted"),
        recall_score(y_test, y_pred, average="weighted"),
        roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted")
    )
