import pandas as pd
import numpy as np
import yaml
import sys
import pickle

from pathlib import Path
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures

params = yaml.safe_load(open('params.yaml'))['featurize']
degree = params['degree']

input_dir = sys.argv[1]
output_dir = sys.argv[2]

Path(output_dir).mkdir(exist_ok=True)

train_file = Path(input_dir) / 'train.csv'
test_file = Path(input_dir) / 'test.csv'

col_names = [
        'age',
        'workclass',
        'weight',
        'education',
        'edu-num',
        'marital-status',
        'occupation',
        'relationship',
        'race',
        'sex',
        'capital-gain',
        'capital-loss',
        'hours-per-week',
        'native-country',
        'class'
]

train_df = pd.read_csv(train_file, sep=',', names=col_names)
test_df = pd.read_csv(test_file, sep=',', names=col_names)

train_df = train_df.apply(LabelEncoder().fit_transform)
test_df = test_df.apply(LabelEncoder().fit_transform)

poly = PolynomialFeatures(degree=degree, interaction_only=True)

train_y = train_df['class']
test_y = test_df['class']

train_df = train_df.drop('class', axis=1)
test_df = test_df.drop('class', axis=1)

train_df = np.column_stack((poly.fit_transform(train_df), train_y))
test_df = np.column_stack((poly.fit_transform(test_df), test_y))

train_output = Path(output_dir) / 'train.p'
test_output = Path(output_dir) / 'test.p'

with open(train_output, 'wb') as f:
    pickle.dump(train_df, f)

with open(test_output, 'wb') as f:
    pickle.dump(test_df, f)

