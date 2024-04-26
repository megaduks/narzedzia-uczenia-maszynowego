# Ludwig

[Ludwig](https://ludwig-ai.github.io/ludwig-docs/) to uniwersalna biblioteka do trenowania modeli uczenia maszynowego, obsługująca kilkanaście rodzajów modeli. Ludwig to wewnętrzna biblioteka Ubera, która została udostępniona jako otwarte oprogramowanie.

Główne cechy Ludwiga to:
- brak konieczności programowania: cały proces trenowania, ewaluacji i inferencji jest realizowany deklaratywnie
- ogólność: narzędzie się konfiguruje samoczynnie na podstawie deklaracji typów danych, więc rozszerzanie funkcjonalności o nowe typy danych jest relatywnie proste
- elastyczność: biblioteka daje możliwość bardzo precyzyjnego kontrolowania procesu treningu, oferując jednocześnie sensownie dobrane wartości domyślne
- rozszerzalność: dodawanie nowych modeli jest proste

Cała praca z Ludwigiem sprowadza się do właściwego spreparowania pliku z danymi (w formacie `csv`) oraz przygotoaniu jednego pliku konfiguracyjnego w formacie `yaml` (dla osób nie zaznajomionych z YAML-em polecam [krótki tutorial](https://www.cloudbees.com/blog/yaml-tutorial-everything-you-need-get-started/)). Najciekawszym pomysłem w Ludwigu jest oparcie przetwarzania na enkoderach i dekoderach, które są związane z konkretnymi typami danych. W chwili obecnej Ludwig obsługuje następujące typy danych:
- binarne
- numeryczne
- kategoryczne
- zbiory
- sekwencje
- tekst
- przebiegi czasowe (timeseries)
- obrazy
- audio
- daty
- wektory

Ideę przetwarzania w Ludwigu obrazuje poniższy diagram:

![ludwig structure](ludwig.png)

Łącząc określony typ wejścia z określonym typem wyjścia otrzymujemy konkretny rodzaj modelu:
- tekst + kategoryczny = klasyfikacja tekstu
- obraz + kategoryczny = klasyfikacja obrazu
- obraz + tekst = opis obrazu (image captioning)
- audio + binarny = weryfikacja mówcy 
- text + sekwencja = NER
- kategoryczny, numeryczny, binarny + numeryczny = regresja
- przebieg czasowy + numeryczny = forecast
- kategoryczny, numeryczny, binarny + binarny = fraud detection

## Klasyfikacja tekstu

Na początek przygotujemy zbiór danych zawierający dwa rodzaje tweetów: tweety dotyczące pandemii COVID-19 oraz tweety ogólne. Naszym zadaniem będzie wytrenowanie modelu który może rozpoznawać tweety o pandemii.

Uruchom trenowanie modelu korzystając z poniższego polecenia:

```bash

ludwig train \
    --dataset data/tweets/tweets.csv \
    --config_str '{input_features: [{name: tweet, type: text}], output_features: [{name: label, type: category}]}'
```

Oczywiście specyfikowanie konfiguracji treningu w linii poleceń szybko staje się kłopotliwe. Utwórz nowy plik `model-tweets.yaml` i umieść w nim poniższą treść:

```yaml
input_features:
    -
        name: tweet
        type: text

output_features:
    -
        name: label
        type: category

training:
    batch_size: 64
    epochs: 5
```

Uruchom trenowanie modelu korzystając z polecenia:

```bash
ludwig train \
    --dataset data/tweets/tweets.csv \
    --config model-tweets.yaml
```

Domyślnym dekoderem dla danych tekstowych jest `parallel_cnn` inspirowany pracą Kima [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882). Załóżmy, że zamiast korzystać z konwolucji na poziomie słów, spróbujemy konwolucji na poziomie pojedynczych znaków. W sekcji `input_features` dodaj klucz `level` z wartością `char` i uruchom ponownie trening.

Być może problemem był nie tyle poziom embeddingu tekstu, co raczej sposób podziału na tokeny (ze względu na charakterystykę tokenów spotykanych w tweetach). Powróć do domyślnego poziomu tokenizacji `word` i dodaj w sekcji `input_features` nowy słownik `preprocessing`, w którym umieść informację o preferowanym podziale przez spacje. Właściwy fragment pliku konfiguracyjnego powinien wyglądać następująco:

```yaml
input_features:
    -
        name: tweet
        type: text
        level: word
        preprocessing:
            word_tokenizer: space
```

Dodajmy jeszcze możliwość modyfikacji parametrów uczenia. W sekcji `training` dodaj jeszcze dwa parametry związane z uczeniem oraz zmień kryterium zatrzymania uczenia:

```yaml
training:
    batch_size: 64
    epochs: 5
    decay: True
    learning_rate: 0.001
    validation_metric: accuracy
```

W tej chwili posługujemy się siecią konwolucyjną do kodowania znaków. Zamiast tego możemy spróbować zaaplikować sieć rekurencyjną. Jeszcze niedawno (tj. przed pojawieniem się architektury transformerów) do przetwarzania tekstu wykorzystywano powszechnie sieci rekurencyjne w architekturze LSTM ze względu na ich zdolność do zapamiętywania kontekstu przetwarzanego tekstu. Zmień sekcję `input_features` pliku konfiguracyjnego w taki sposób, żeby wykorzystać enkoder RNN LSTM.

```yaml
input_features:
    -
        name: tweet
        type: text
        level: word
        encoder: rnn
        cell_type: lstm
        preprocessing:
            word_tokenizer: space
```

Ludwig zawiera większość najnowszych modeli językowych dostępnych w module `huggingface`. Pełna lista enkoderów dla tekstu jest dostępna [tutaj](https://ludwig-ai.github.io/ludwig-docs/user_guide/#text-input-features-and-encoders). Na koniec zobaczmy, jak z zadaniem poradzi sobie BERT. Zmień rodzaj enkodera na `bert`.

## Praca z klasycznymi zbiorami danych

Do ilustracji sposobu wykorzystania Ludwiga do klasycznego problemu klasyfikacji posłużymy się znanym zbiorem opisującym pasażerki/ów Titanica. Obejrzyj zbiory danych `data/titanic/train.csv` i `data/titanic/test.csv`.

Następnie przygotuj plik konfiguracyjny `model-titanic.yaml` o następującej postaci.

```yaml
input_features:
    -
        name: Pclass
        type: category
    -
        name: Name
        type: text
    -
        name: Sex
        type: category
    -
        name: Age
        type: numerical
        preprocessing:
          missing_value_strategy: fill_with_mean
    -
        name: SibSp
        type: numerical
    -
        name: Parch
        type: numerical
    -
        name: Ticket
        type: category
    -
        name: Fare
        type: numerical
        preprocessing:
          missing_value_strategy: fill_with_mean
    -
        name: Cabin
        type: category
    -
        name: Embarked
        type: category

output_features:
    -
        name: Survived
        type: binary
```

Uruchom trening modelu wykonując polecenie:

```bash
ludwig train \
    --dataset data/titanic/train.csv \
    --config model-titanic.yaml
```

Spróbujmy dokonać niewielkich modyfikacji w definicji modelu:

- dla atrybutu `Pclass` zmień sposób kodowania na one-hot (`encoder: sparse`)
- zmień rodzaj atrybutu `Sex` na `binary`
- usuń informację o porcie zaokrętowania

a następnie uruchom trening ponownie, tym razem jawnie wskazując miejsce zapisania modelu:

```bash
ludwig train \
    --dataset data/titanic/train.csv \
    --config model-titanic.yaml \
    --output_directory results/titanic
```

W kolejnym kroku przetestujemy model korzystając z polecenia `experiment`. Zanim uruchomisz poniższe polecenie, dodaj do pliku konfiguracyjnego ograniczenie treningu do 10 epok.

```bash
ludwig experiment \
    --k_fold 5 \ 
    --dataset data/titanic/train.csv \
    --config model-titanic.yaml
```

Obejrzyj wyniki eksperymentu.

W następnym kroku zwizualizujemy proces uczenia.

```bash
ludwig visualize \
    --visualization learning_curves \
    --training_statistics results/titanic/experiment_run/training_statistics.json \
    --file_format pdf \
    --output_directory results/titanic/
```

Wyniki wizualizacji zostały zapisane na dysku w kontenerze, więc musimy jeszcze przekopiować je na komputer-host w celu obejrzenia. Sprawdź identyfikator kontenera i przekopiuj pliki.

```bash
sudo docker ps

sudo docker cp <container_ID>:/home/results/titanic/learning_curves_Survived_accuracy.df .
sudo docker cp <container_ID>:/home/results/titanic/learning_curves_Survived_loss.df .
sudo docker cp <container_ID>:/home/results/titanic/learning_curves_combined_loss.df .
```

## Klasyfikacja obrazów

Do przedstawienia sposobu pracy z obrazami wykorzystamy prosty problem klasyfikacji kształtów. W zbiorze danych masz przygotowane zdjęcia okręgów, trójkątów i kwadratów. Przygotuj plik `model-images.yaml` z następującą definicją modelu:

```yaml
input_features:
    -
        name: image_path
        type: image
        encoder: stacked_cnn
        preprocessing:
            resize_method: crop_or_pad
            width: 128
            height: 128

output_features:
    -
        name: label
        type: category

training:
    batch_size: 8
    epochs: 25
```

a następnie uruchom trening modelu:

```bash
ludwig train \
    --dataset image-train.csv \
    --config model-images.yaml \
    --output_directory results/images/
```

Obejrzyj wynik procesu uczenia (wskaż właściwy dla siebie katalog ze statystykami treningu)

```bash
ludwig visualize \
    --visualization learning_curves \
    --training_statistics results/images/<run>/training_statistics.json
```

W następnym kroku zaaplikujemy wytrenowany model do nowego zbioru danych.

```bash
ludwig predict \
    --dataset image-test.csv \
    --model_path results/images/<run>/model/ \
    --output_directory results/images
```

Obejrzyj wyniki zaaplikowania modelu:

```bash
cat results/images/label_predictions.csv

cat results/images/label_probabilities.csv
```

Korzystając z linii poleceń możemy łatwo połączyć pliki i sprawdzić, które przykłady zostały błędnie sklasyfikowane.

```bash
paste image-test.csv results/images/label_predictions.csv
```

## Serwowanie modelu

Ludwig udostępnia też prosty mechanizm, dzięki któremu możemy uruchomić model jako usługę. W tym celu musimy jeszcze doinstalować parę zależności:

```bash
pip install ludwig[serve]
```

a następnie uruchomić serwer:

```bash
ludwig serve \
    --model_path results/images/experiment_run/model 
    --port 8081 \
    --host 0.0.0.0
```

Po uruchomieniu serwisu można wysyłać do niego żądania:

```bash
curl http://localhost:8081/predict -X POST -F 'image_path=@data/shapes/serve/triangle.png'
```

## Dostęp przez API

Oczywiście cała funkcjonalność Ludwiga jest też dostępna poprzez API. Poniższy przykład pokazuje, w jaki sposób można tego dokonać. 


```python
from ludwig.api import LudwigModel
import pandas as pd

df = pd.read_csv('data/tweets/tweets.csv')

model_definition = {
    'input_features':[
        {'name':'tweet', 'type':'text'},
    ],
    'output_features': [
        {'name': 'label', 'type': 'category'}
    ],
    'training':
        {'epochs': 100, 'batch_size': 32, 'learning_rate': 0.01},
}

model = LudwigModel(model_definition, logging_level=25)
model.train(dataset=df)

tweets = [
    {'tweet': 'I just had my vaccine shot today!'},
    {'tweet': 'Trump claims serious voter fraud in Nevada'},
    {'tweet': 'EU stops the administration of the Pfizer vaccine'}
]

model.predict(dataset=tweets, data_format='dict')
```

## Zadanie samodzielne

Zapoznaj się z opisem zbioru danych zawierającego [tweety o amerykańskich liniach lotniczych](https://www.kaggle.com/crowdflower/twitter-airline-sentiment?select=Tweets.csv). Zbiór ten masz dostępny w katalogu `data/airlines/tweets.csv`.

Spróbuj samodzielnie wytrenować jeden z poniższych modeli:

- model sentymentu: przewiduje ogólny wydźwięk tweeta (negative, neutral, positive) na podstawie tekstu tweeta
- model klasyfikacji: przewiduje jakiej linii lotniczej dotyczy dany tweet
- model regresji: przewiduje liczbę re-twetów jakie zdobędzie dany tweet
