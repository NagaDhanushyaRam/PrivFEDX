"""SHAP hello-world: explain a logistic regression on toy data."""
import numpy as np
import shap
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

X, y = make_classification(n_samples=200, n_features=6, n_informative=3, random_state=0)
model = LogisticRegression(max_iter=1000).fit(X, y)

explainer = shap.LinearExplainer(model, X)
sv = explainer.shap_values(X[:5])
print(f"SHAP OK | shap {shap.__version__} | values shape {np.shape(sv)}")