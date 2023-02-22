import pandas as pd
import sklearn
import yaml
import random
import sys

from pathlib import Path
from sklearn.model_selection import train_test_split

params = yaml.safe_load(open('params.yaml'))['prepare']

split = params['split']
random.seed(params['seed'])

input_file = Path(sys.argv[1])
train_output = Path('data') / 'prepared' / 'train.csv'
test_output = Path('data') / 'prepared' / 'test.csv'

Path('data/prepared').mkdir(parents=True, exist_ok=True)

df = pd.read_csv(input_file, sep=',')
train_df, test_df = train_test_split(df, train_size=split)

train_df.to_csv(train_output, index=None)
test_df.to_csv(test_output, index=None)
