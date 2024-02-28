# Data Versioning Control (DVC)

## Wprowadzenie

Celem laboratorium jest zapoznanie studentów z podstawami wykorzystania narzędzia `dvc` do wersjonowania danych i modeli w projektach wykorzystujących uczenie maszynowe. 

Najprostszym sposobem jest zainstalowanie biblioteki wewnątrz środowiska wirtualnego lub korzystając z condy, choć możliwe jest też zainstalowanie bezpośrednio z repozytorium lub pakietu. Szczegóły dotyczące instalacji można znaleźć na [stronie porjektu](https://dvc.org/doc/install/linux)

Wewnątrz uruchomionego kontenera znajdujesz się w katalogu `home`. To będzie domyślny katalog do realizacji ćwiczenia. 

Pierwszym krokiem jest zainicjalizowanie w katalogu projektu środowiska `git` a następnie zainicjalizowanie środowiska `dvc`. Ponieważ znajdujesz się wewnątrz kontenera, konieczne jest ustawienie globalnej nazwy użytkownika dla `git`.


```bash
git config --global user.email "you@example.com"
git config --global user.name "Your Name"

git init
dvc init
git status
```
Zapoznaj się z utworzonymi przez `dvc` plikami i zapisz aktualny stan repozytorium w początkowym commicie.

```bash
git add .dvc/config
git add .dvc/.gitignore
git add .dvcignore

git commit -m "feat: initialize dvc repo"
```

## Wersjonowanie danych

Głównym celem `dvc` jest umożliwienie wersjonowania dużych plików z danymi, których wykorzystanie wewnątrz `git` [jest problematyczne](https://docs.github.com/en/github/managing-large-files/working-with-large-files). W poniższym przykładzie wykorzystamy `dvc` do tego, żeby mieć możliwość pracowania z różnymi wersjami tego samego pliku.

Przed rozpoczęciem pracy zapoznaj się z plikami `data/adult.names` i `data/adult.data`. Dodaj oba pliki do repozytorium, aby `dvc` zaczęło śledzić zmiany w tych plikach. 

```bash
dvc add data/adult.data
dvc add data/adult.names
```

Obejrzyjmy pliki, które powstały w wyniku dodania plików danych do repozytorium


```bash
cat data/adult.data.dvc
cat data/adult.names.dvc
```

W celu umożliwienia śledzenia zmian w plikach danych konieczne jest dodanie utworzonych właśnie plików `*.dvc` oraz plku `data/.gitignore` do repozytorium `git`.


```bash
git add data/.gitignore data/adult.data.dvc data/adult.names.dvc
git commit -m "feat: add ADULT dataset to dvc"
```

Kolejnym krokiem będzie utworzenie zewnętrznego repozytorium danych. DVC obsługuje wiele usług składawania danych, takich jak Amazon S3, Google Cloud Storage, zewnętrzne dyski dostępne przez `ssh`, systemy HDFS, i wiele innych. My posłużymy się alternatywnym lokalnym katalogiem do symulowania zewnętrznego repo, a następnie wypchniemy do niego zmiany.


```bash
mkdir -p ~/dvcrepo
dvc remote add -d repozytorium ~/dvcrepo
git commit .dvc/config -m "feat: add local dir as remote dvc repo"
```
Wypchnij lokalne pliki danych do "zdalnego" repozytorium.

```bash
dvc push
tree ~/dvcrepo/
cat ~/dvcrepo/1a/7cdb3ff7a1b709968b1c7a11def63e
```

Zewnętrzne repozytorium może być wykorzystane do ściągnięcia danych w przypadku dokonania niechcianych zmian, odtworzenia gałęzi eksperymentu, itd.


```bash
rm -rf .dvc/cache/
rm data/adult.data
rm data/adult.names

tree data/
```
Jak widać, przez przypadek usunęliśmy pliki z danymi. Dzięki `dvc` łatwo odtworzymy stan repozytorium.

```bash
dvc pull

tree data/
```

W kolejnym kroku dokonamy zmian w pliku danych, usuwając wszystkie linie dotyczące pracowników federalnych. Sprawdźmy, ile jest takich rekordów, a następnie je usuniemy.


```bash
cat data/adult.data | wc -l
grep 'Federal-gov' data/adult.data | wc -l
```

```bash
sed -i "/Federal-gov/d" data/adult.data
cat data/adult.data | wc -l
```
Zmianę wprowadzoną do pliku danych należy zarejestrować w `dvc`.

```bash
dvc add data/adult.data
git commit data/adult.data.dvc -m "feat: remove federal workers"

dvc push
```

Jeśli chcemy cofnąć tę zmianę, to konieczne jest cofnięcie się do właściwej wersji pliku `adult.data.dvc` i wykonanie komendy `dvc checkout` w celu zsynchronizowania danych.


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

## Dostęp do zewnętrznych repozytoriów danych

Mając skonfigurowane repozytorium `git` korzystające z `dvc` możemy z łatwością wykorzystać `dvc` to szybkiego pobierania danych i modeli, współdzielenia danych z innymi, itp. Wynik poprzedniego rozdziału umieściłem w repozytorium https://github.com/megaduks/dvc-tutorial i teraz zobaczymy, w jaki sposób można wyświetlić listę wersjonowanych danych i z nimi pracować.


```bash
dvc list https://github.com/megaduks/dvc-tutorial data
```

Te zbiory danych możemy ściągnąć jednym poleceniem, np. do nowego projektu


```bash
mkdir nowy_projekt
cd nowy_projekt
dvc get https://github.com/megaduks/dvc-tutorial data
tree .
```

Niestety, w ten sposób straciliśmy informację o źródle plików danych, nie mamy możliwości ich ponownego powiązania. Komenda `dvc get` jest w pewnym sensie odpowiednikiem `wget`. Jeśli chcemy zachować połączenie między danymi i ich źródłem, konieczne jest wykonanie komendy `dvc import`.


```bash
mkdir -p nowszy_projekt/data
dvc import https://github.com/megaduks/dvc-tutorial/ data/adult.data \
    -o nowszy_projekt/data/adult.data
```


```bash
cat nowszy_projekt/data/adult.data.dvc
```

Jak widać, metadane związane z plikiem `adult.data` zawierają teraz informację o tym, z jakiego zewnętrznego repozytorium plik pochodzi wraz ze szczegółowymi hashami informującymi o wersji pliku danych. Dodatkowo, możemy bardzo łatwo śledzić zmiany pliku z danymi w zewnętrznym repozytorium.


```bash
dvc update nowszy_projekt/data/adult.data.dvc
```

Interesującą możliwością jest też dostęp programistyczny do plików umieszczonych w zewnętrznych repozytoriach.


```python
import dvc.api

with dvc.api.open('data/adult.data', repo='https://github.com/megaduks/dvc-tutorial') as f:
    for _ in range(10):
        print(f.readline())
```

## Potoki danych

Najciekawszą możliwością oferowaną przez `dvc` jest możliwość zarządzania reprodukowalnymi potokami przetwarzania. Zilustrujemy to na przykładzie, w ramach którego wykonamy następujące czynności:

- przygotujemy dane poprzez usunięcie części rekordów
- dodanmy nowy atrybut 
- wytrenujemy model
- zbadamy dokładność modelu

Oczywiście poszczególne etapy będą maksymalnie proste, bo naszym celem jest zbudowanie potoku przetwarzania danych. 

Utworzymy pierwszy krok potoku w ramach którego wczytamy plik tekstowy i zamienimy na wersję spiklowaną. Utwórz plik `params.yaml` i umieść w nim następującą zawartość

```
prepare:
  split: 0.75
  seed: 42
```

Następnie przygotuj plik `prepare.py` i wypełnij go poniższą treścią:


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

Teraz możemy utworzyć pierwszy przepływ w którym:

- utworzymy nazwany krok (`-n prepare`)
- przekażemy parametry (`-p prepare.seed,prepare.split`)
- przekażemy zależności (`-d src/prepare.py -d data/adult.data`)
- wskażemy wyjście z kroku (`-o data/prepared/`)
- uruchomimy skrypt i przekażemy dane wejściowe


```bash
dvc stage add -n prepare \
    -p prepare.seed,prepare.split \
    -d src/prepare.py -d data/adult.data \
    -o data/prepared \
    python src/prepare.py data/adult.data
dvc repro
```

W wyniku tego kroku pojawiły się pliki wynikowe oraz specjalny plik `dvc.yaml` pokazujący w sposób przyjazny dla użytkownika konfigurację całego przepływu.


```bash
cat dvc.yaml
tree data/prepared/
```

Dobrym pomysłem po utworzeniu pierwszego kroku przetwarzania będzie zapisanie go w `git`.

```bash
git add dvc.yaml dvc.lock data/.gitignore params.yaml
git commit -m "feat: create preparation step" 
```

Kolejnym krokiem jest dodanie do przepływu zadania polegającego na przetransformowaniu danych. Dokonamy rekodowania wszystkich atrybutów kategorycznych przy użyciu [klasy LabelEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html#sklearn.preprocessing.LabelEncoder) oraz wyznaczymy interakcje między atrybutami korzystając z [klasy PolynomialFeatures](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html#sklearn.preprocessing.PolynomialFeatures). Ta ostatnia klasa wykorzysta parametr `degree`. Zaktualizuj plik parametrów żeby wyglądał następująco:

```
prepare:
  split: 0.75
  seed: 42
featurize:
  degree: 2
```

Następnie utwórz plik `featurize.py` i wypełnij go poniższą treścią.


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

Możemy zatem dokonać inżynierii cech uruchamiając polecenie


```bash
dvc stage add -n featurize \
    -p featurize.degree \
    -d src/featurize.py -d data/prepared/ \
    -o data/features \
    python src/featurize.py data/prepared/ data/features/
```

Aby nie utracić efektów pracy powinniśmy dotychczasowe kroki potoku zapisać w repozytorium `git`.


```bash
git add data/.gitignore dvc.lock dvc.yaml params.yaml
git commit -m 'feat: create featurization step'
```

Następnym krokiem jest uruchomienie treningu. Posłużymy się tu prostym skryptem który trenuje las losowy, korzystając z dwóch parametrów: maksymalnej głębokości drzewa i liczby drzew wchodzących w skład lasu losowego. Zmodyfikuj plik parametrów aby wyglądał następująco:

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

Utwórz plik `train.py` i umieść w nim poniższy kod:


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

Jak widać, skrypt oczekuje dwóch parametrów podanych z linii poleceń (katalog z danymi wejściowymi i katalog w którym należy umieścić wyniki skryptu). Dodanie kroku do przepływu wymaga wykonania następującego polecenia:


```bash
dvc stage add -n train \
    -p train.max_depth,train.n_estimators \
    -d src/train.py -d data/features/ \
    -o data/models/ \
    python src/train.py data/features/ data/models/
```

Tradycyjnie już zapisujemy zmiany w przepływie w repozytorium `git`


```bash
git add data/.gitignore dvc.lock dvc.yaml params.yaml
git commit -m 'feat: create training step'
```

Po co w ogóle tworzyliśmy plik `dvc.yaml`? Na pierwszy rzut oka wydaje się to wszystko nadmiernie skomplikowane. Ale tu właśnie ujawnia się główna zaleta `dvc`, umożliwia pełną reprodukowalność całych przepływów za pomocą jednego polecenia.


```bash
dvc repro
```

Zmień jakiś parametr w sekcji `train` (np. zwiększ liczbę drzew składowych) i ponownie uruchom cały przepływ. Które etapy zostały uruchomione? Zmień parametr w sekcji `prepare` (np. sposób podziału na zbiór uczący i testujący) i znów uruchom przepływ. Co stało się tym razem?

Aktualny przepływ możesz też zwizualizaować przy pomocy polecenia `dvc dag`.

## Eksperymenty

Ostatnim elementem `dvc` jaki zobaczymy będzie uruchomienie eksperymentów. Zanim jednak przejdziemy do eksperymentów, przygotujemy w pliku `evaluate.py` kod do oceny jakości nauczonego modelu. Utwórz taki plik i umieść w nim poniższy kod.


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

Tym razem dodanie kroku ewaluacji do potoku będzie bardziej skomplikowane, ponieważ musimy też uwzględnić specjalny plik do przechowywania wartości metryk oraz plik przechowywania danych na potrzeby wykresów. 


```bash
dvc stage add -n evaluate \
    -d src/evaluate.py -d data/models/ -d data/features/ \
    -M scores.json \
    --plots-no-cache prc.json \
    python src/evaluate.py data/models/ data/features/ scores.json prc.json
```

Spójrzmy, jak wygląda ostatecznie plik opisujący cały przepływ.


```bash
cat dvc.yaml
```

Nie zapominamy o zapisaniu aktualnego stanu przepływu w repozytorium `git`.


```bash
git add dvc.lock dvc.yaml 
git commit -m 'feat: create evaluation step'
```

W wyniku wykonania przepływu powstał plik `scores.json` zawierający jedyną aktualnie wykorzystywaną miarę AUROC.


```bash
cat scores.json
```

W pliku `prc.json` z kolei zapisane są informacje o procesie uczenia (*precision-recall curve*). Zapiszmy oba te pliki do repozytorium.


```bash
git add scores.json prc.json
git commit -m 'feat: add evaluation metrics'
```

Spróbujmy uruchomić eksperyment przy zmienionych parametrach i sprawdźmy, jak to wpłynie na naszą miarę. Zmień parametr `degree` na wartość 3 oraz zwiększ liczbę `n_estimators` do 25. Następnie uruchom przepływ.


```bash
dvc repro
dvc params diff
dvc metrics diff

dvc plots diff -x recall -y precision
```

Plik z wykresem został utworzony wewnątrz kontenera. Jeśli chcesz go obejrzeć, najprościej jest przekopiować ten plik na hosta.

```bash
docker ps
docker cp <container-id>:/home/dvc_plots/index.html .
```
