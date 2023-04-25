# Adnotacja danych z Prodigy

## Rozpoznawanie nazwanych encji

1. Prodigy jest narzędziem silnie związanym z biblioteką `spacy`. Wzorce rozpoznawania nazwanych encji definiuje w sposób analogiczny do mechanizmu matcherów obecnego w `spacy`.

2. Normalnie zaczęlibyśmy od zbudowania pliku zawierającego kilka przykładów tego, jak wygląda nazwa języka programowania. Wykorzystamy do tego format spopularyzowany przez bibliotekę `spacy`.

```
{"label": "PROG_LANG", "pattern": [{"lower": "java"}]}
{"label": "PROG_LANG", "pattern": [{"lower": "c"}, {"lower": "+"}, {"lower": "+"}]}
{"label": "PROG_LANG", "pattern": [{"lower": "objective"}, {"lower": "c"}]}
...
```

3. Przygotowanie takiego pliku jest jednak pracochłonne. Zamiast tego pozwolimy, żeby Prodigy zbudowało dla nas ten plik na podstawie niewielkiego zbioru początkowych terminów.

```bash
prodigy terms.teach language_names en_core_web_lg --seeds python,julia,prolog,lisp,java,smalltalk,go
```

4. Po wygenerowaniu odpowiedniej liczby przykładów możemy obejrzeć wynik tej anotacji

```bash
prodigy db-out language_names
```

5. Ten zbiór danych powinien zostać jeszcze przetłumaczony na format zgodny ze SpaCy

```bash
prodigy terms.to-patterns language_names --label PROG_LANG --spacy-model blank:en > ./language_names.jsonl

head ./language_names.jsonl
```

3. Mając tak przygotowany plik możemy uruchomić ręczną adnotację danych z komentarzy.

```bash
prodigy ner.manual programming_languages en_core_web_lg ./programming.jsonl.bz2 --loader reddit --label PROG_LANG --patterns language_names.jsonl
```

4. Po zaanotowaniu odpowiedniej liczby przypadków możemy obejrzeć adnotacje

```bash
prodigy print-dataset programming_languages
```

5. Istnieje też możliwość wyeksportowania adnotowanego zbioru danych 

```bash
prodigy db-out programming_languages
```

6. Kolejnym krokiem jest wytrenowanie początkowego modelu NER. To jest dopiero początek, później poprawimy jakość tego modelu.

```bash
prodigy train /tmp/initial --ner programming_languages --base-model en_core_web_lg --eval-split 0.2 --training.eval_frequency 100 --training.patience 1000
```

7. W kolejnym kroku sprawdzimy, jak dobrze radzi sobie nasz model NER. Poprosimy model o adnotację i będziemy jedynie binarnie poprawiać decyzje modelu, co jest oczywiście dużo łatwiejszą i mniej męczącą pracą. Wyłączymy też z anotacji przypadki już wcześniej przez nas anotowane.

```bash
prodigy ner.correct programming_languages_corrected /tmp/initial/model-best ./programming.jsonl.bz2 --loader reddit --label PROG_LANG --exclude programming_languages
```

8. Ostatnim krokiem jest połączenie obu zbiorów danych i wytrenowanie ostatecznego modelu rozpoznawania encji

```bash
prodigy train /tmp/final --ner programming_languages,programming_languages_corrected --base-model en_core_web_lg --eval-split 0.2 --training.eval_frequency 100 --training.patience 1000
```

10. Wypróbujmy utworzony model 

```python
import spacy

nlp = spacy.load('/tmp/final')

doc = nlp('My favourite programming languages are Python, C++ and Scheme')

for e in doc.ents:
    print(e.text, e.label_, e.start, e.end)
```

## Adnotacja obrazów

1. Spróbujmy teraz wykorzystać Prodigy do ręcznego oznaczenia interesujących nas elementów na zdjęciach twarzy.

```bash
prodigy image.manual faces_dataset ./images --label MOUTH,EYES
```

2. Możemy także uruchomić proces klasyfikacji zdjęć. W najprostszej wersji adnotujemy zdjęcia w sposób binarny, odpowiadając na proste pytanie: czy zdjęcie przedstawia osobę dorosłą?

```bash
prodigy mark adult_child_image ./images --loader  images --label ADULT --view-id classification
```

3. Jeśli chcemy wykorzystać Prodigy do klasyfikacji zdjęć w przypadku gdy liczba klas jest większa niż 2, musimy przygotować swoją własną "receptę" wykorzystującą interfejs `choice`. Sprowadza się to do udekorowania funkcji, która jest generatorem zwracającym odpowiednio sformatowane słowniki.

```python
import prodigy
from prodigy.components.loaders import Images

OPTIONS = [
    {"id": 1, "text": "SERIOUS"},
    {"id": 2, "text": "SAD"},
    {"id": 3, "text": "GLAD"},
]

@prodigy.recipe("classify-images")
def classify_images(dataset, source):
    def get_stream():
        stream = Images(source)
        for example in stream:
            example["options"] = OPTIONS
            yield example

    return {
        "dataset": dataset,
        "stream": get_stream(),
        "view_id": "choice",
        "config": {
            "choice_style": "single",
            "choice_auto_accept": True
        }
    }
```

```bash
prodigy classify-images emotions_dataset ./images -F recipe.py
```

## Klasyfikacja tekstu

Klasyfikacja tekstu jest bardzo podobna do trenowania modelu NER, z tą różnicą, że ocenie podlega cały dokument (lub jego zdania)

1. W pierwszym kroku musimy ręcznie anotować pewną liczbę komentarzy aby model był w stanie znaleźć związek między słowami występującymi w tekście a etykietą tekstu.

```bash
prodigy textcat.manual programming_comment programming.jsonl.bz2 --loader reddit --label PROGRAMMING,OTHER --exclusive
```

```bash
prodigy print-dataset programming_comment
```

2. Drugim krokiem jest uruchomienie treningu modelu na przygotowanych adnotacjach

```bash
prodigy train /tmp/textcat --textcat programming_comment --training.eval_frequency 100 --training.patience 1000
```

3. Wytrenowany model możemy łatwo sprawdzić w działaniu

```python
import spacy

nlp = spacy.load('/tmp/textcat')

doc = nlp('Java has just released version 15')

for cat in doc.cats:
    print(f'label {cat}: {doc.cats[cat]}')
```

## Zadanie samodzielne

Wykorzystaj plik `homebrewing.jsonl.bz2` do wytrenowania własnego modelu NER do rozpoznawania gatunków piwa (APA, IPA, Vermont, pilsner, lager, weizen, bock, helles, ...)

