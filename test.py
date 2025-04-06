import numpy as np
from sklearn.datasets import make_regression, make_classification
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split
from MSBoost import MSBoostRegressor, MSBoostClassifier

def regression_test():
    X, y = make_regression(n_samples=10000, n_features=10, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    regressor = MSBoostRegressor(n_estimators=100, learning_rate=0.1, return_vals=True, bayes=True)
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print("MSBoost Regression Test - Mean Squared Error:", mse)
    print(regressor.ensemble)
    iterations = list(range(1, len(regressor.errors) + 1))

    plt.figure(figsize=(10, 6))
    plt.plot(iterations, regressor.errors, marker='.', linestyle='-', color='blue', label='Error')
    plt.title('MSBoostRegressor Error per Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def classification_test():
    # Generate synthetic binary classification data
    X, y = make_classification(n_samples=10000, n_features=10, n_informative=8, n_redundant=0, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    
    # Initialize and fit the classifier
    classifier = MSBoostClassifier(n_estimators=100, learning_rate=0.1, return_vals=True, bayes=True)
    classifier.fit(X_train, y_train)

    # Predict and evaluate using accuracy
    y_pred = classifier.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("MSBoost Classification Test - Accuracy:", acc)
    print(classifier.ensemble)

    iterations = list(range(1, len(classifier.errors) + 1))

    plt.figure(figsize=(10, 6))
    plt.plot(iterations, classifier.errors, marker='.', linestyle='-', color='blue', label='Error')
    plt.title('MSBoostClassifier Error per Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    regression_test()
    classification_test()
