import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import mlflow

# 1. Load data
df = pd.read_csv("data/churn.csv")

# Fix column issue
df = df.rename(columns=lambda x: x.strip())

# 2. Preprocessing
df = df.dropna()

# Convert categorical to numeric
df = pd.get_dummies(df, drop_first=True)

# Target column (important fix)
target_col = [col for col in df.columns if "Churn" in col][-1]

y = df[target_col]
X = df.drop([target_col], axis=1)

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 4. MLflow tracking
mlflow.set_experiment("churn_prediction")

with mlflow.start_run():

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)

    # Log parameters
    mlflow.log_param("model", "RandomForest")
    mlflow.log_param("n_estimators", 100)

    # Log metrics
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)

    print("Accuracy:", acc)
    print("Precision:", precision)
    print("Recall:", recall)