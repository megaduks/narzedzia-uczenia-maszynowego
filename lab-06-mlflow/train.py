import plac

import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet

from pathlib import Path

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient


@plac.opt('input_file', 'Input file with training data', Path, 'i')
@plac.opt('alpha', 'Alpha parameter for ElasticNet', float, 'a')
@plac.opt('l1_ratio', 'L1 ratio parameter for ElasticNet', float, 'l')
def main(input_file: Path, alpha: float=0.5, l1_ratio: float=0.5):

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

        mlflow.sklearn.log_model('lr', 'model')

if __name__ == "__main__":
    plac.call(main)
