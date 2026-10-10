import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Part 1 — Prepare the Dataset
print("Loading California Housing dataset...")
data = fetch_california_housing()
X = data.data
y = data.target

# Split into 80% Training, 20% Validation 
# We use random_state=42 to make the split reproducible
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Configurations for the three runs specified in the task
experiments = [
    {"run_name": "Run 1", "max_depth": 3, "learning_rate": 0.1},
    {"run_name": "Run 2", "max_depth": 5, "learning_rate": 0.05},
    {"run_name": "Run 3", "max_depth": 7, "learning_rate": 0.01},
]

# Set the experiment name in MLflow
mlflow.set_experiment("House_Price_Prediction")

# Part 2 & 3 — Train Multiple Experiments and Log the Model
for exp in experiments:
    # Start an MLflow run for each configuration
    with mlflow.start_run(run_name=exp["run_name"]):
        print(f"Training {exp['run_name']} (max_depth={exp['max_depth']}, learning_rate={exp['learning_rate']})...")
        
        # 1. Train the model
        model = GradientBoostingRegressor(
            max_depth=exp["max_depth"],
            learning_rate=exp["learning_rate"],
            random_state=42 # Fixed seed for reproducible training
        )
        model.fit(X_train, y_train)
        
        # 2. Calculate the validation metrics
        predictions = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        # 3. Log the parameters
        mlflow.log_param("max_depth", exp["max_depth"])
        mlflow.log_param("learning_rate", exp["learning_rate"])
        
        # 4. Log the metrics
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        
        # 5. Save the trained model as an MLflow artifact
        mlflow.sklearn.log_model(model, "model", skops_trusted_types=["sklearn.tree._tree.Tree"])
        
        print(f"{exp['run_name']} completed -> RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}\n")

print("All experiments finished! Run 'mlflow ui' in your terminal to compare the models.")