import numpy as np
from sklearn.datasets import make_regression, make_classification
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split
from MSBoost import MSBoostRegressor, MSBoostClassifier

def regression_test():
    # Generate synthetic regression data
    X, y = make_regression(n_samples=50, n_features=5, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    
    # Initialize and fit the regressor
    regressor = MSBoostRegressor(n_estimators=100, learning_rate=0.001, return_vals=True)
    regressor.fit(X_train, y_train)
    
    # Predict and evaluate using mean squared error
    y_pred = regressor.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print("Regression Test - Mean Squared Error:", mse)

def classification_test():
    # Generate synthetic binary classification data
    X, y = make_classification(n_samples=50, n_features=5, n_informative=5, n_redundant=0, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    
    # Initialize and fit the classifier
    classifier = MSBoostClassifier(n_estimators=100, learning_rate=0.001, return_vals=True)
    classifier.fit(X_train, y_train)
    
    # Predict and evaluate using accuracy
    y_pred = classifier.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("Classification Test - Accuracy:", acc)

if __name__ == '__main__':
    regression_test()
    classification_test()
