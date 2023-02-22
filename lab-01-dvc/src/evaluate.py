import sys
import os
import pickle
import json

from sklearn.metrics import precision_recall_curve
import sklearn.metrics as metrics
from pathlib import Path

model_file = Path(sys.argv[1]) / 'model.p'
test_file = Path(sys.argv[2]) / 'test.p'

scores_file = sys.argv[3]
plots_file = sys.argv[4]

with open(model_file, 'rb') as f:
    model = pickle.load(f)

with open(test_file, 'rb') as f:
    test_df = pickle.load(f)

X = test_df[:,:-1]
y = test_df[:,-1]

predictions_by_class = model.predict_proba(X)
y_pred = predictions_by_class[:, 1]

precision, recall, thresholds = precision_recall_curve(y, y_pred)
auc = metrics.auc(recall, precision)

with open(scores_file, 'w') as f:
    json.dump({'auc': auc}, f)

with open(plots_file, 'w') as f:
    json.dump({'prc': [{
            'precision': p,
            'recall': r,
            'threshold': t
        } for p, r, t in zip(precision, recall, thresholds)
    ]}, f)
