import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn
from src.data_processing import get_model_ready_data # Imports your completed Task 3 & 4 logic

def train_and_log_model(model, X_train, y_train, X_test, y_test, model_name, params=None):
    """Trains a model, evaluates it, and logs results to MLflow."""
    
    # Start MLflow run to track this experiment
    with mlflow.start_run(run_name=f"{model_name}_Run"):
        print(f"Starting MLflow run for {model_name}...")

        # Hyperparameter Tuning using GridSearchCV
        if params:
            # GridSearch to find the best C parameter for Logistic Regression or n_estimators/max_depth for GB
            grid_search = GridSearchCV(model, params, cv=3, scoring='roc_auc', n_jobs=-1)
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            mlflow.log_params(grid_search.best_params_)
        else:
            best_model = model
            best_model.fit(X_train, y_train)

        # Predict and Evaluate
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]
        y_pred = best_model.predict(X_test)
        
        # Calculate Metrics
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        # Log Metrics
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        
        print(f"   ROC-AUC: {roc_auc:.4f}")

        # Log Model Artifact
        # Register the model to the MLflow Model Registry (Crucial for Task 6 Deployment)
        mlflow.sklearn.log_model(
            best_model, 
            "model", 
            registered_model_name=f"CreditRisk_{model_name}",
            # Include input example for deployment clarity
            input_example=X_test.iloc[[0]].to_dict('list') 
        )
        
        return roc_auc

def main():
    # Set up MLflow tracking
    mlflow.set_tracking_uri("sqlite:///mlruns.db") # Uses a local SQLite database file for tracking
    mlflow.set_experiment("Credit_Risk_Model_Final_Submission")

    # 1. Get Model-Ready Data (Runs src/data_processing.py)
    X, y = get_model_ready_data()
    if X is None:
        return

    # 2. Data Splitting
    # Stratify ensures the high-risk target ratio is maintained in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Define and Train Hyperparameters for Logistic Regression (Baseline Model)
    lr_params = {'C': [0.01, 0.1, 1]} # C is the regularization inverse strength
    lr_model = LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced')
    train_and_log_model(lr_model, X_train, y_train, X_test, y_test, "LogisticRegression", lr_params)

    # 4. Define and Train Hyperparameters for Gradient Boosting Classifier (Stronger Model)
    gb_params = {'n_estimators': [50, 100], 'max_depth': [3]}
    gb_model = GradientBoostingClassifier(random_state=42)
    train_and_log_model(gb_model, X_train, y_train, X_test, y_test, "GradientBoosting", gb_params)

if __name__ == "__main__":
    main()