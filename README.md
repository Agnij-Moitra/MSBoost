# MSBoost: Using Model Selection with Multiple Base Estimators for Gradient Boosting
## Abstract
Gradient boosting is a widely used machine learning algorithm for tabular regression, classification and ranking. Although, most of the open source implementations of gradient boosting such as XGBoost, LightGBM and others have used decision trees as the sole base estimator for gradient boosting. This paper, for the first time, takes an alternative path of not just relying on a static base estimator (usually decision tree), and rather trains a list of models in parallel on the residual errors of the previous layer and then selects the model with the least validation error as the base estimator for a particular layer. This paper has achieved state-of-the-art results when compared to other gradient boosting implementations on 50+ tabular regression and classification datasets. Furthermore, ablation studies show that MSBoost is particularly effective for small and noisy datasets. Thereby, it has a significant social impact especially in tabular machine learning problems in the domains where it is not feasible to obtain large high quality datasets. 

## Build Cython Version:
```
python setup.py build_ext --inplace

```