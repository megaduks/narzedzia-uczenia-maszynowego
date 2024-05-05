import typer
from typing_extensions import Annotated

import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet

from pathlib import Path

import mlflow
import mlflow.sklearn

from mlflow.tracking import MlflowClient

def autolog(run):

    tags = {
        k: v
        for k, v in run.data.tags.items()
        if not k.startswith("mlflow.")
    }

    artifacts = [
        f.path
        for f
        in MlflowClient().list_artifacts(run.info.run_id, "model")
    ]

    print(f"run_id: {run.info.run_id}")
    print(f"artifacts: {artifacts}")
    print(f"params: {run.data.params}")
    print(f"metrics: {run.data.metrics}")
    print(f"tags: {tags}")

def main(
        input_file: Annotated[Path, typer.Option("--input_file", "-i", help="Input file with training data")],
        alpha: Annotated[float, typer.Option("--alpha", "-a", help="Alpha param for ElasticNet")] = 0.5,
        l1_ratio: Annotated[float, typer.Option("--l1_ratio", "-l", help="L1 ratio param for ElasticNet")] = 0.5
        ):


    assert input_file, "Please provide a file with the training data"

    df = pd.read_csv(input_file, sep=';')

    df_train, df_test = train_test_split(df, train_size=0.8)

    X_train = df_train.drop(['quality'], axis=1)
    X_test = df_test.drop(['quality'], axis=1)
    y_train = df_train['quality']
    y_test = df_test['quality']

    with mlflow.start_run():
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(X_train, y_train)

        y_pred = lr.predict(X_test)

        rmse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2score = r2_score(y_test, y_pred)

        print(f"ElasticNet(alpha={alpha},l1_ratio={l1_ratio}): RMSE={rmse}, MAE={mae}, R2={r2score}")

        mlflow.log_param('alpha', alpha)
        mlflow.log_param('l1_ratio', l1_ratio)
        mlflow.log_metric('rmse', rmse)
        mlflow.log_metric('mae', mae)
        mlflow.log_metric('r2score', r2score)

        # mlflow.sklearn.log_model('lr', 'model')

    # mlflow.sklearn.autolog()
    # lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)

    # with mlflow.start_run() as run:
    #     lr.fit(X_train, y_train)

    # autolog(mlflow.get_run(run_id=run.info.run_id))

if __name__ == "__main__":
    typer.run(main)
