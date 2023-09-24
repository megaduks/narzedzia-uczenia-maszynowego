# Data Versioning Control (DVC)

## Introduction

The aim of this laboratory is to familiarize the students with the basics of `dvc`, a tool for versioning data and models in projects using machine learning.

The easiest way to install the library  is inside a virtual environment or using conda, although it is also possible to install directly from the official repository or package. Installation details can be found on the [project website](https://dvc.org/doc/install/linux).

Inside the running container, you are in the `home` directory. This will be the default directory for the exercise.

The first step is to initialize a `git` repository in the project directory and then initialize a `dvc` repository. Since you are inside a container, it is necessary to set a global username for `git`.


```bash
git config --global user.email "you@example.com"
git config --global user.name "Your Name"

git init
dvc init
git status
```

Review the files created by `dvc` and save the current state of the repository in the initial commit.

```bash
git add .dvc/config
git add .dvc/.gitignore
git add .dvcignore

git commit -m "feat: initialize dvc repo"
```

## Data versioning

The main purpose of `dvc` is to enable versioning of large data files that are problematic to use inside `git`(https://docs.github.com/en/github/managing-large-files/working-with-large-files). In the example below, we will use `dvc` to be able to work with different versions of the same file.

Before you start, review the files `data/adult.names` and `data/adult.data`. Add both files to the repository so that `dvc` starts tracking changes of these files.

```bash
dvc add data/adult.data
dvc add data/adult.names
```

Let's look at the additional files that are created by `dvc` when adding data files to the repository:

```bash
cat data/adult.data.dvc
cat data/adult.names.dvc
```

In order to be able to track changes in the data files, it is necessary to add the `*.dvc` files and `data/.gitignore` to the `git` repository.


```bash
git add data/.gitignore data/adult.data.dvc data/adult.names.dvc
git commit -m "feat: add ADULT dataset to dvc"
```

The next step will be to create an external data repository. DVC supports many storage services such as Amazon S3, Google Cloud Storage, external drives accessed via `ssh`, HDFS systems, and many others. We will use an alternative local directory to simulate an external repo and then push the changes to it.

```bash
mkdir -p ~/dvcrepo
dvc remote add -d repozytorium ~/dvcrepo
git commit .dvc/config -m "feat: add local dir as remote dvc repo"
```

Push your local data files to a "remote" repository.

```bash
dvc push
tree ~/dvcrepo/
cat ~/dvcrepo/1a/7cdb3ff7a1b709968b1c7a11def63e
```

An external repository can be used to pull data in case of unwanted changes, re-creating an experiment branch, etc.

```bash
rm -rf .dvc/cache/
rm data/adult.data
rm data/adult.names

tree data/
```

As you can see, we "accidentally" deleted the data files. Thanks to `dvc` we can easily restore the state of the repository.

```bash
dvc pull

tree data/
```

In the next step, we will make changes to the data file in order to remove all lines related to federal employees. Let's check how many such records there are and then delete them.


```bash
cat data/adult.data | wc -l
grep 'Federal-gov' data/adult.data | wc -l
```

```bash
sed -i "/Federal-gov/d" data/adult.data
cat data/adult.data | wc -l
```

The changes made to the data file must be added in `dvc`.

```bash
dvc add data/adult.data
git commit data/adult.data.dvc -m "feat: remove federal workers"

dvc push
```

If we want to undo this change, it is necessary to roll back to the correct version of the `adult.data.dvc` file using `git` and execute the `dvc checkout` command to synchronize the data.


```bash
git log
```


```bash
git checkout 34685237371f63dc2fa2f997ce9f2aa514c0ffe9 data/adult.data.dvc
dvc checkout
```

```bash
grep 'Federal-gov' data/adult.data
```

```bash
git commit data/adult.data.dvc -m "feat: revert deletion of federal workers"
```

## Access to external data repositories

Once we configured a `dvc` environment jointly with  `git`, we can easily use `dvc` to quickly download data and models, share data with others, etc. The result of the previous part has been placed in the repository https://github.com/megaduks/dvc-tutorial and now we will see how to display a list of versioned data and work with it.



```bash
dvc list https://github.com/megaduks/dvc-tutorial data
```

We can download these data sets with one command, e.g. to a new project

```bash
mkdir new_project
cd new_project
dvc get https://github.com/megaduks/dvc-tutorial data
tree .
```

Unfortunately, doing it this way, we lose the information and history about the source, we have no way to re-link them. The `dvc get` command is somewhat equivalent to `wget`. To maintain the relation between the data and its source, it is necessary to execute the `dvc import` command.


```bash
mkdir -p new_project/data
dvc import https://github.com/megaduks/dvc-tutorial/ data/adult.data \
    -o new_project/data/adult.data
```


```bash
cat new_project/data/adult.data.dvc
```

As you can see, the metadata associated with the file `adult.data` now includes the information about which external repository the file comes from, along with the hashes that provide information about the version of the data file. Additionally, we can very easily track changes of the data file from the external repository.

```bash
dvc update new_project/data/adult.data.dvc
```

Another interesting possibility is to programmatically access to files located in external repositories.

```python
import dvc.api

with dvc.api.open('data/adult.data', repo='https://github.com/megaduks/dvc-tutorial') as f:
    for _ in range(10):
        print(f.readline())
```

## Data pipelines

The most interesting feature offered by `dvc` is the ability to manage reproducible processing pipelines. We will illustrate this with an example in which we will do the following:

- we will prepare the data by splitting the records into training and test data
- we will add a new attribute
- we will train the model
- we will test the accuracy of the model

We will create the first step of the pipeline in which we will load a text file and replace it with a spliced version. Create a file `params.yaml` and place the following content in it:


```yaml
prepare:
  split: 0.75
  seed: 42
```

Then create the file `prepare.py` and add the following content

```python
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

train_df.to_csv(train_output, header=None)
test_df.to_csv(test_output, header=None)
```

Now we can create the first stage of our pipeline as follow:

- we will create a stage with a name (`-n prepare`)
- we will pass some parameters (`-p prepare.seed,prepare.split`)
- we will pass some dependencies (`-d src/prepare.py -d data/adult.data`)
- we will indicate the stage output(`-o data/prepared/`)
- we will run the script using `dvc repro`

```bash
dvc stage add -n prepare \
    -p prepare.seed,prepare.split \
    -d src/prepare.py -d data/adult.data \
    -o data/prepared \
    python src/prepare.py data/adult.data
dvc repro
```

As a result, the output files have been created and `dvc` created a special file `dvc.yaml` which shows the configuration of the entire pipeline in a user-friendly way.

```bash
cat dvc.yaml
tree data/prepared/
```

Once you've created your first processing step, it's a good idea to save it using `git`.

```bash
git add dvc.yaml dvc.lock data/.gitignore params.yaml
git commit -m "feat: create preparation step" 
```

The next step is to add a data transformation stage to the pipeline. We will encode all categorical attributes using [`LabelEncoder` class](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html#sklearn.preprocessing.LabelEncoder) and determine the interactions between the attributes using [the class `PolynomialFeatures`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html#sklearn.preprocessing.PolynomialFeatures). We will add the `degree` parameter of the `PolynomialFeatures` class as parameter to the stage. Update the parameter file to look like this:

```
prepare:
  split: 0.75
  seed: 42
featurize:
  degree: 2
```

Then create a `featurize.py` file and add the following content.

```python
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

```

We can add the  feature engineering stage by running a command

```bash
dvc stage add -n featurize \
    -p featurize.degree \
    -d src/featurize.py -d data/prepared/ \
    -o data/features \
    python src/featurize.py data/prepared/ data/features/
```

In order not to lose the results of our work, we should save the current steps of the pipeline using `git`.


```bash
git add data/.gitignore dvc.lock dvc.yaml params.yaml
git commit -m 'feat: create featurization step'
```

The next step is to train a model. We will use a simple script that trains a random forest using two parameters: the maximum tree depth and the number of trees included in the random forest. Modify the parameter file to look like this:

```
prepare:
  split: 0.75
  seed: 42
featurize:
  degree: 2
train:
  max_depth: 2
  n_estimators: 5
```

Create a `train.py` file and place the following code in it:


```python
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

```

As you can see, the script expects two parameters provided from the command line (the directory with the input data and the directory in which the script results should be placed). As usually, add the step to the pipeline usingthe following command:

```bash
dvc stage add -n train \
    -p train.max_depth,train.n_estimators \
    -d src/train.py -d data/features/ \
    -o data/models/ \
    python src/train.py data/features/ data/models/
```

Again, we save changes to the pipepline in the `git` repository:

```bash
git add data/.gitignore dvc.lock dvc.yaml params.yaml
git commit -m 'feat: create training step'
```

At this stage (no pun intended), you might wonder why we create the `dvc.yaml` file at all? At first, it all seems overly complicated. But this is where the main advantage of `dvc` comes into play: it allows full reproducibility of entire pipelines with a single command.

```bash
dvc repro
```

Change some of the parameters in the `train` section (e.g. increase the number of component trees) and run the entire pipeline again. Which stages have been started? Change the parameter in the `prepare` section (e.g. the method of splitting into training and testing sets) and run the pipeline again. What happened this time?

You can also visualize the current pipeline using the `dvc dag` command..

## Experiments

The last element of `dvc` that we will see are the experiments. However, before we move on to experiments, we will prepare a new stageto assess the quality of the trained models. Create a file `evaluate.py` and place the following code in it.



```python
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
```

This time, adding an evaluation stage to the pipeline will be more complicated because we also need to include a special file to store the metric values and a data storage file for the plots.

```bash
dvc stage add -n evaluate \
    -d src/evaluate.py -d data/models/ -d data/features/ \
    -M scores.json \
    --plots-no-cache prc.json \
    python src/evaluate.py data/models/ data/features/ scores.json prc.json
```

Let's look at what the final file describing the entire pipeline looks like.


```bash
cat dvc.yaml
```

Do not forget to save the current state of the pipeline in the `git` repository.

```bash
git add dvc.lock dvc.yaml 
git commit -m 'feat: create evaluation step'
```

As a result of executing the pipeline, a `scores.json` file was created containing the only metric that we have configured aka the AUROC.

```bash
cat scores.json
```

The `prc.json` file contains information about the *precision-recall curve*. Let's save both of these files to the repository.

```bash
git add scores.json prc.json
git commit -m 'feat: add evaluation metrics'
```

Let's try running the experiment with different input parameters and see how it affects our evaluation metric. Change the `degree` parameter to 3 and increase the number of `n_estimators` to 25. Then run the pipeline again.

```bash
dvc repro
dvc params diff
dvc metrics diff

dvc plots diff -x recall -y precision
```

The plot file was created inside the container. If you want to view it, the easiest way is to copy the file to your host system.

```bash
docker ps
docker cp <container-id>:/home/dvc_plots/index.html .
```