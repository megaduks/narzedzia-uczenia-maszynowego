# MLflow

[MLflow](https://mlflow.org) to biblioteka która obsługuje wiele elementów procesu MLOps, w tym:

- śledzenie różnych wersji eksperymentów (_MLflow Tracking_)
- wdrożenie modelu ML w postaci artefaktu do wielokrotnego użycia i współdzielenia (_MLflow Projects_)
- zarządzanie modelami zbudowanymi w oparciu o różne biblioteki (_MLflow Projects_)
- udostępnianie centralnego repozytorium modeli wraz z zarządzaniem ich cyklem życia (_MLflow Project Repository_)

Jedną z ciekawych cech MLflow jest fakt, że biblioteka jest całkowicie agnostyczna względem bibliotek do tworzenia modeli ML. Cała funkcjonalność jest dostępna z poziomu REST API oraz jako zbiór komend linii poleceń, istnieją też API do Pythona, Javy, oraz R.

Fundamentalnym pojęciem w ramach `mlflow` jest **artefakt**. Jest to dowolny plik lub katalog związany z projektem, przechowywany w zewnętrznym repozytorium. Artefakty mogą być logowane w repozytorium, mogą być też pobierane i zapisywane do repozytorium. Artefakty mogą być obiektami na dysku loklanym, ale mogą to też być pliki przechowywane na S3, w HDFS, modele wraz z wersjami, itp.

Scenariusze użycia `mlflow` obejmują:

- pracę indywidualnych badaczy i inżynierów: możliwość śledzenia treningu na lokalnych maszynach, utrzymywanie wielu wersji konfiguracji, wygodne przechowywanie modeli przygotowanych w różnych architekturach
- pracę zespołów _data science_: możliwość porównywania wyników różnych algorytmów, unifikacja terminologii (nazwy skryptów i parametrów), współdzielenie modeli
- pracę dużych organizacji: współdzielenie i wielokrotne użycie modeli i projektów, wymiana wiedzy, ułatwienie produktyzacji procesów
- MLOps: możliwość wdrażania modeli z różnych bibliotek jako prostych plików w systemie operacyjnym
- badaczy: możliwość współdzielenia i uruchamiania repozytoriów GitHub

W poniższym przykładzie zbudujemy model regresji liniowej przewidującej jakość wina. 

### Śledzenie przebiegu eksperymentu

Rekordy opisujące poszczególne uruchomienia (_runs_) mogą być przechowywane:
- w lokalnym katalogu
- w bazie danych (MySQL, SQLite, PostgreSQL)
- na serwerze HTTP z uruchomionym MLFlow
- w przestrzeni pracy Databricks

Stwórz plik `train.py` i zamieść w nim poniższy kod:


```python
import plac

import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet

from pathlib import Path

import mlflow
import mlflow.sklearn

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

        mlflow.sklearn.log_model(lr, 'model')


if __name__ == "__main__":
    plac.call(main)
```

Obejrzyj zawartość pliku z danymi

```bash
bash$ cat winequality.csv
```

Sprawdź poprawność funkcjonowania skryptu uruchamiając go z linii poleceń

```bash
bash$ python train.py --help

bash$ python train.py --input-file data/winequality.csv --alpha 0.4 --l1-ratio 0.75
```

Obejrzyj strukturę katalogu `mlruns`

```bash
bash$ tree mlruns
```

Uruchom kilka razy trening przekazując różne wartości parametrów

Uruchom serwer MLflow i obejrzyj informacje zgromadzone o przebiegu eksperymentu pod adresem [localhost:5000](http://localhost:5000)

```bash
bash$ mlflow ui -p 5000 -h 0.0.0.0
```

Wróć do linii poleceń i uruchom serię eksperymentów, sprawdzając różne kombinacje parametrów. Wcześniej upewnij się, że ustawienia językowe terminala są angielskie (przecinek dziesiętny powoduje błąd)

```bash
bash$ export LANG=en_US

bash$ for a in $(seq 0.1 0.1 1.0)
    do
        for l in $(seq 0.1 0.1 1.0)
        do
            python train.py -i data/winequality.csv -a $a -l $l
        done
    done
```

Ponownie uruchom serwer `mlflow` i obejrzyj wyniki. Zaznacz wszystkie przebiegi i dodaj do porównania. Sprawdź dostępne wizualizacje poszczególnych miar.

Zmodyfikuj plik `train.py` dodając, po logowaniu metryk, logowanie modelu. W tym celu dopisz poniższą linię:

```python
mlflow.sklearn.log_metric(lr, 'model')
```

a następnie uruchom jednorazowo trening i obejrzyj wyniki. Zobacz, w jakiej postaci model został zapisany w repozytorium.

Dodaj w kodzie fragment powodujący przypisanie tagu do danego eksperymentu i zaobserwuj tagi w repozytorium.

```python

# set a single tag
mlflow.set_tag('version','0.9')

# set a list of tags
mlflow.set_tags({
    'author': 'Mikolaj Morzy',
    'date': '01.01.2021',
    'release': 'candidate'
})
```

Sprawdź, jak jest zapisywane w repozytorium niepoprawne uruchomienie treningu. W tym celu uruchom skrypt z błędnym wywołaniem parametrów.

### Automatyczne śledzenie parametrów i metryk

`mlflow` potrafi w sposób automatyczny śledzić wartości parametrów i metryk dla wielu popularnych bibliotek ML. Zmień zawartość pliku `train.py` w następujący sposób:

- dodaj import klasy `MlflowClient` z modułu `mlflow.tracking`
- zakomentuj cały kod uruchomiony w kontekście `mlflow.start_run()`
- bezpośrednio po podziale na zbiór uczący i testujący dodaj ponizszy kod:

```python
mlflow.sklearn.autolog()
lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)

with mlflow.start_run() as run:
    lr.fit(X_train, y_train)

autolog(mlflow.get_run(run_id=run.info.run_id))
```

- dodaj wyżej w kodzie poniższą funkcję:

```python
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
```

Uruchom ponownie kod treningu i zaobserwuj wynik.

### Grupowanie przebiegów w eksperymenty

`mlflow` pozwala na grupowanie wielu przebiegów w postaci nazwanych eksperymentów. Pierwszym krokiem jest stworzenie nazwanego eksperymentu. Wykonaj poniższą komendę:

```bash
bash$ mlflow experiments create --experiment-name simple-regression
```

Powtórz wcześniejsze ćwiczenie, wcześniej ustawiając zawartość zmiennej środowiskowej `MLFLOW_EXPERIMENT_NAME`

```bash
bash$ export MLFLOW_EXPERIMENT_NAME=simple-regression

bash$ python train.py -i data/winequality.csv
bash$ python train.py -i data/winequality.csv -a 0.1
bash$ python train.py -i data/winequality.csv -l 0.9
```

i obejrzyj wynik w repozytorium.

Alternatywą dla użycia zmiennej środowiskowej jest przekazanie parametru `--experiment_name` w momencie wywołania polecenia `mlflow experiment run`.

### Pakowanie modelu

W następnym kroku zbudujemy całą paczkę zawierającą kod trenujący prosty model. Utwórz katalog `regression` i stwórz w nim dwa pliki: `MLProject` oraz `conda.yaml`.

Plik `MLProject` zawiera definicję projektu MLflow. Umieść w nim następującą treść:

```
name: linear_regression_example

conda_env: conda.yaml

entry_points:
    main:
        command: "python train.py"
```

Plik `conda.yaml` zawiera definicję środowiska w którym będzie uruchomiony kod.

```yaml
name: regression-example
channels:
  - defaults
  - anaconda
  - conda-forge
dependencies:
  - python=3.7
  - scikit-learn
  - pip
  - pip:
    - mlflow>=1.2333
```

Plik `train.py` zawiera kod treningu modelu. W tym przypadku jest to bardzo prosty kod trenujący klasyfikator na zabawkowym przykładzie.

```python
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
```

Uruchom paczkę wydając poniższe polecenie i wcześniej wskazując na lokalizację Condy:

```bash
export MLFLOW_CONDA_HOME=/path/to/local/conda

bash$ mlflow run regression
``` 

### Uruchomienie modelu bezpośrednio z repozytorium

W następnym kroku zapiszemy tę paczkę jako repozytorium Git i uruchomimy eksperyment bezpośrednio z repozytorium

- utwórz zdalne repozytorium na GitHubie (np. o nazwie `mlflow_example`
- wejdź do katalogu `regression` i zainicjalizuj repozytorium komendą `git init`
- dodaj zawartość katalogu do repozytorium komendą `git add .`
- stwórz pierwszy commit komendą `git commit -m "MLflow experiment repo created"`
- skopiuj URL zdalnego repozytorium
- wykonaj poniższe komendy

```bash
bash$ git remote add origin <url zdalnego repozytorium>
bash$ git remote -v
```
- wypchnij lokalne zmiany do zdalnego repozytorium komendą `git push origin master`
- uruchom eksperyment w zdalnym repozytorium komendą `mlflow run <url zdalnego repozytorium>`

### Pakowanie modelu z parametrami

Jako następny przykład utworzymy eksperyment, który wymaga podania parametrów. Utwórz katalog `wine-quality` i stwórz w nim dwa pliki: `MLProject` oraz `conda.yaml`.

W pliku `MLProject` umieść następującą treść:

```
name: wine_quality_model
conda_env: conda.yaml

entry_points:
    main:
        parameters:
            input_file: {type: str}
            alpha: {type: float, default=0.5}
            l1_ratio: {type: float, default=0.5}
        command: "python train.py -i {input_file} -a {alpha} -l {l1_ratio}"
```

W pliku `conda.yaml` umieść następującą treść:

```yaml
name: wine_quality_model
channels:
    - defaults
dependencies:
    - python=3.7
    - pip
    - pip:
        - sklearn>=0.23.2
        - mlflow>=1.23
```

Przekopiuj do katalogu `wine-quality` także pliki `train.py` i plik z danymi (w poniższym przykładzie jego nazwę zmieniono na `data.csv`).

Ustaw zmienną `MLFLOW_CONDA_HOME` tak aby wskazywała Twoją instalacje Condy. Uruchom utworzenie gotowej paczki z modelem, danymi i zależnościami.

```bash
bash$ mlflow run wine-quality -P input_file=data.csv -P alpha=0.12 -P l1_ratio=0.79
```

### Serwowanie zbudowanego modelu

Utworzony model może zostać z łatwością wdrożony. Obejrzyj jeszcze raz w repozytorium meta-dane przebiegu w którym zalogowano także model. Zwróć uwagę na obecność dwóch plików: spiklowanego modelu oraz pliku tekstowego z meta-danymi. Przeczytaj meta-dane i odnotuj identyfikator przebiegu (`run_id`)

Uruchom serwowanie modelu, instalując pakiet `pyenv` i wydając polecenie

```bash
bash$ curl https://pyenv.run | bash
bash$ export PATH=$HOME/.pyenv/bin:$PATH

bash$ mlflow models serve -m /path/to/model/subfoler -p 5000 -h 0.0.0.0
```

Korzystając z REST API dokonaj predykcji wydając polecenie

```bash
bash$ curl -X POST -H "Content-Type:application/json; format=pandas-split" --data '{"columns":["alcohol", "chlorides", "citric acid", "density", "fixed acidity", "free sulfur dioxide", "pH", "residual sugar", "sulphates", "total sulfur dioxide", "volatile acidity"],"data":[[12.8, 0.029, 0.48, 0.98, 6.2, 29, 3.33, 1.2, 0.39, 75, 0.66]]}' http://localhost:5000/invocations
```

### Zadanie samodzielne

Pobierz zbiór danych z [World Happiness Report 2021](https://www.kaggle.com/ajaypalsinghlo/world-happiness-report-2021). 

Korzystając z narzędzia `mlflow` przygotuj paczkę zawierającą kod do trenowania modelu. Zbuduj model [drzewa decyzyjnego](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html) i postaraj się przeprowadzić eksperyment który pozwoli Ci wybrać najlepsze wartości parametrów:

- maksymalna głębokość drzewa
- miara oceny punktu podziału
- minimalna liczba instancji w liściu

W eksperymencie posłuż się metrykami średniego błędu bezwględnego, średniego błędu kwadratowego, oraz współczynnika determinacji R2.

Po wytreniowaniu modelu przygotuj paczkę zawierającą cały kod i udostępnij model do predykcji przy użyciu protokołu REST API.
