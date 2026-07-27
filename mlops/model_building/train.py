# for data manipulation
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
# for model serialization
import joblib
# for creating a folder
import os
# for hugging face authentication to upload files
from huggingface_hub import HfApi
from huggingface_hub.utils import RepositoryNotFoundError
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("mlops-training-experiment")

hf_token = os.getenv("HF_TOKEN", "").strip()
if not hf_token:
    raise RuntimeError("HF_TOKEN is required to upload the trained model.")

api = HfApi(token=hf_token)

Xtrain_path = "hf://datasets/krish2105/bank-customer-churn/Xtrain.csv"
Xtest_path = "hf://datasets/krish2105/bank-customer-churn/Xtest.csv"
ytrain_path = "hf://datasets/krish2105/bank-customer-churn/ytrain.csv"
ytest_path = "hf://datasets/krish2105/bank-customer-churn/ytest.csv"

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path)
ytest = pd.read_csv(ytest_path)

numeric_features = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
]

categorical_features = ["Geography"]

# Flatten the single-column target frames before fitting and computing class weights.
ytrain_series = ytrain.squeeze("columns")
ytest_series = ytest.squeeze("columns")
class_weight = ytrain_series.value_counts().iloc[0] / ytrain_series.value_counts().iloc[1]

preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features),
)

xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42)

param_grid = {
    "xgbclassifier__n_estimators": [50, 75, 100, 125, 150],
    "xgbclassifier__max_depth": [2, 3, 4],
    "xgbclassifier__colsample_bytree": [0.4, 0.5, 0.6],
    "xgbclassifier__colsample_bylevel": [0.4, 0.5, 0.6],
    "xgbclassifier__learning_rate": [0.01, 0.05, 0.1],
    "xgbclassifier__reg_lambda": [0.4, 0.5, 0.6],
}

model_pipeline = make_pipeline(preprocessor, xgb_model)

with mlflow.start_run():
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(Xtrain, ytrain_series)

    results = grid_search.cv_results_
    for i in range(len(results["params"])):
        with mlflow.start_run(nested=True):
            mlflow.log_params(results["params"][i])
            mlflow.log_metric("mean_test_score", results["mean_test_score"][i])
            mlflow.log_metric("std_test_score", results["std_test_score"][i])

    mlflow.log_params(grid_search.best_params_)
    best_model = grid_search.best_estimator_
    classification_threshold = 0.45

    y_pred_train = (
        best_model.predict_proba(Xtrain)[:, 1] >= classification_threshold
    ).astype(int)
    y_pred_test = (
        best_model.predict_proba(Xtest)[:, 1] >= classification_threshold
    ).astype(int)

    train_report = classification_report(ytrain_series, y_pred_train, output_dict=True)
    test_report = classification_report(ytest_series, y_pred_test, output_dict=True)

    mlflow.log_metrics(
        {
            "train_accuracy": train_report["accuracy"],
            "train_precision": train_report["1"]["precision"],
            "train_recall": train_report["1"]["recall"],
            "train_f1-score": train_report["1"]["f1-score"],
            "test_accuracy": test_report["accuracy"],
            "test_precision": test_report["1"]["precision"],
            "test_recall": test_report["1"]["recall"],
            "test_f1-score": test_report["1"]["f1-score"],
        }
    )

    model_path = "best_churn_model.joblib"
    joblib.dump(best_model, model_path)
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved as artifact at: {model_path}")

    repo_id = "krish2105/churn-model"
    repo_type = "model"

    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Model repository '{repo_id}' already exists.")
    except RepositoryNotFoundError:
        print(f"Model repository '{repo_id}' not found. Creating it...")
        api.create_repo(
            repo_id=repo_id,
            repo_type=repo_type,
            private=False,
            exist_ok=True,
        )

    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo=model_path,
        repo_id=repo_id,
        repo_type=repo_type,
    )
