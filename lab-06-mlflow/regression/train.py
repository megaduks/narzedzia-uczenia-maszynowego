import numpy as np
from sklearn.linear_model import LinearRegression

import mlflow
import mlflow.sklearn

if __name__ == "__main__":

    X = np.arange(-100,100).reshape(-1, 1)
    y = X**2

    lr = LinearRegression()
    lr.fit(X, y)

    score = lr.score(X, y)

    print(f"Score: {score}")

    mlflow.log_metric("score", score)
    mlflow.sklearn.log_model(lr, "model")

    print(f"Model saved in run {mlflow.active_run().info.run_uuid}")
