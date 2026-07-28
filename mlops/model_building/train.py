import pandas as pd
import numpy as np

# Scikit-Learn 1.6+ compatibility patch for pickled XGBClassifier models
try:
    from sklearn.base import ClassifierMixin
    from sklearn.utils._tags import Tags, TargetTags, ClassifierTags
    ClassifierMixin.__sklearn_tags__ = lambda self: Tags(
        estimator_type='classifier',
        target_tags=TargetTags(required=False),
        transformer_tags=None,
        regressor_tags=None,
        classifier_tags=ClassifierTags()
    )
except Exception:
    pass

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score
import joblib
import os
import mlflow

try:
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("mlops-training-experiment")
except Exception as mlflow_err:
    print(f"MLflow info: {mlflow_err}")

Xtrain = pd.read_csv("Xtrain.csv")
Xtest = pd.read_csv("Xtest.csv")
ytrain = pd.read_csv("ytrain.csv").values.ravel()
ytest = pd.read_csv("ytest.csv").values.ravel()

numeric_features = [
    'Tenure', 'WarehouseToHome', 'HourSpendOnApp', 'NumberOfDeviceRegistered',
    'SatisfactionScore', 'Complain', 'OrderAmountHikeFromlastYear',
    'DaySinceLastOrder', 'CashBackAmount', 'CityTier'
]

categorical_features = [
    'PreferredPaymentMode', 'Gender', 'PreferedOrderCat', 'MaritalStatus'
]

class_weight = float((len(ytrain) - sum(ytrain)) / max(1, sum(ytrain)))

preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
)

xgb_clf = xgb.XGBClassifier(
    scale_pos_weight=class_weight,
    eval_metric='logloss',
    random_state=42
)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', xgb_clf)
])

param_grid = {
    'classifier__n_estimators': [50, 100],
    'classifier__max_depth': [3, 5],
    'classifier__learning_rate': [0.05, 0.1],
}

print("Starting hyperparameter tuning...")
grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='roc_auc', n_jobs=1)
grid_search.fit(Xtrain, ytrain)

best_model = grid_search.best_estimator_
ypred = best_model.predict(Xtest)
ypred_prob = best_model.predict_proba(Xtest)[:, 1]

acc = accuracy_score(ytest, ypred)
rec = recall_score(ytest, ypred)
auc = roc_auc_score(ytest, ypred_prob)

print(f"Best Hyperparameters: {grid_search.best_params_}")
print(f"Test Accuracy: {acc:.4f}, Test Recall: {rec:.4f}, Test ROC-AUC: {auc:.4f}")

joblib.dump(best_model, "best_churn_model.joblib")
print("Model successfully saved to best_churn_model.joblib")
