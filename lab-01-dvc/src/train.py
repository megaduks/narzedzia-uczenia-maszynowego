import sys
import yaml
import pickle

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

params = yaml.safe_load(open('params.yaml'))['train']
max_depth = params['max_depth']
n_estimators = params['n_estimators']

input_dir = sys.argv[1]
output_dir = sys.argv[2]

Path(output_dir).mkdir(exist_ok=True)

train_file = Path(input_dir) / 'train.p'
model_file = Path(output_dir) / 'model.p'

with open(train_file, 'rb') as f:
    train_df = pickle.load(f)

X = train_df[:, :-1]
y = train_df[:, -1]

clf = RandomForestClassifier(
    n_estimators=n_estimators,
    max_depth=max_depth
)
clf.fit(X, y)

with open(model_file, 'wb') as f:
    pickle.dump(clf, f)

