# Streamlit

Streamlit to biblioteka która zamienia skrypty w Pythonie w bogate aplikacje webowe z zaawansowanymi kontrolkami. Biblioteka jest elastyczna i umożliwia tworzenie własnych komponentów rozszerzających bogaty zestaw wbudowanych kontrolek. Najlepszym zastosowaniem dla biblioteki Streamlit jest szybkie prototypowanie  aplikacji wykorzystujących modele statystyczne.

W celu zainstalowania biblioteki wystarczy wydać polecenie

```bash
bash$ pip install streamlit
```

# Podstawowe komponenty

Utwórz osobny katalog i umieść w nim plik `helloworld.py`. W pliku umieść następujący kod:


```python
import streamlit as st

st.title("Hello world app")
st.header("This is my first Streamlit app")

st.write("Hello, world!")
```

W linii poleceń przejdź do katalogu z plikiem i uruchom serwer przy pomocy polecenia

```bash
bash$ streamlit run helloword.py
```

Otwórz wskazany adres w przeglądarce.

Wywołanie funkcji `st.write()` jest nadmiarowe, Streamlit domyślnie wysyła do tej funkcji każdą zmienną którą napotka w skrypcie. Zobacz, co się stanie, gdy do pliku dodasz poniższą linię:


```python
_str = "Hello, universe!"
_str
```

Jeżeli chcesz w prosty i szybki sposób dodać formatowaną zawartość, najłatwiej jest posłużyć się kodem markdown. Dodaj do dokumentu poniższą linię.


```python
st.markdown("> Streamlit is awsome!")
st.markdown("*Mikołaj Morzy*")
st.markdown("[Don't click on me](https://theuselessweb.com/)")
```

Analogicznie można dołączyć do dokumentu fragment w Latexu


```python
st.latex("\LaTeX: e^{i\pi}+1=0")
```

Biblioteka Streamlit jest w sposób szczególny przystosowana do pracy z danymi przechowywanymi w obiektach biblioteki `pandas`. Zaimportuj tę bibliotekę, wczytaj plik danych i wyświetl jego zawartość w aplikacji.


```python
import pandas as pd

df = pd.read_csv("titanic.csv")

df
```

Zamiast korzystać z dynamiczego wyświetlania obiektu `DataFrame` można go też wyświetlić w sposób statyczny.


```python
st.table(df.head())
```

Streamlit posiada dedykowany komponent do wyświetlania danych tabelarycznych z możliwością dynamicznego dostosowania listy kolumn. Zamień powyższą linię na:


```python
cols = ["Name", "Sex", "Age"]

df_multi = st.multiselect("Columns", df.columns.tolist(), default=cols)
st.dataframe(df[df_multi])
```

Dane, które już są umieszczone w obiekcie `DataFrame` mogą posłużyć do wizualizacji. Przygotujmy dane w taki sposób, aby móc wyświetlić rozkład wieku pasażerek/ów Titanica.


```python
age_distribution = df.Age.dropna().value_counts()

st.bar_chart(age_distribution)
```

Jeśli chcesz włączyć wyświetlanie fragmentu aplikacji warunkowo, możesz to łatwo zrobić dzięki komponentowi `st.checkbox`. Zamień ostatnie dwie linie na poniższy kod:


```python
if st.checkbox("Show age distribution?"):
    age_distribution = df.Age.dropna().value_counts()

    st.chart(age_distribution)
```

Inną możliwością modyfikowania sposobu wyświetlania danych jest posłużenie się komponentem `st.selectbox` w celu wybrania jednej z przedstawionych opcji. Dodaj do pliku poniższy kod:


```python
display_sex = st.selectbox("Select sex to display", df.Sex)

st.dataframe(df[df.Sex == display_sex])
```

Duża liczba kontrolek w głównym panelu wyświetlania danych może źle wpłynąć na czytelność aplikacji. Dowolny komponent może być automatycznie przeniesiony do paska bocznego poprzez zamianę wywołania `st.komponent` na `st.sidebar.komponent`. Spróbuj przenieść listę wyboru płci do wyświetlenia do paska bocznego. 

Kod wyświetlany w głównym panelu nie musi zajmować całej szerokości panelu. Panel może być podzielony na dowolną liczbę kolumn przy użyciu komponentu `st.columns`. 

Umieść w pliku poniższy kod i zaobserwuj jego działanie.


```python
left_column, right_column = st.columns(2)

button_clicked = left_column.button("Click me!")

if button_clicked:
    right_column.write("Thank you!")
```

Szczególnie długie opisy mogą być umieszczone w komponencie `st.expander`:


```python
import lorem

expander = st.expander("Lorem ipsum")
expander.write(lorem.paragraph())
```

Jeśli skrypt zawiera w sobie jakąś długą operację, jej postępo może być łatwo raportowany przy użyciu komponentu `st.progress`. Przeanalizuj poniższy przykład, zwróć szczególną uwagę na użycie komponentu `st.empty` jako tzw. _placeholdera_.


```python
import time

"Here we begin a long computation"

current_iteration = st.empty()
progress_bar = st.progress(0)

for i, _ in enumerate(range(100)):
    current_iteration.text(f"Iteration {i+1}")
    progress_bar.progress(i+1)
    time.sleep(0.1)
    
"Finally, computation is completed."
```

# Komponenty zaawansowane

W poniższym przykładzie wygenerujemy fikcyjne zamówienia Ubera na terenie Poznania i zapoznamy się z bardziej zaawansowanymi sposobami wizualizacji danych. Utwórz nowy plik `uber_app.py` i umieść w nim następujący kod:


```python
import streamlit as st
import pandas as pd
import numpy as np
import random

from datetime import datetime, timedelta

st.title("Uber pickups in Poznan")

def generate_data(n_rows: int = 100) -> pd.DataFrame:
    end = datetime.now()
    start = end - timedelta(days=100)
    
    dates = [
        random.random() * (end - start) + start 
        for _ in range(n_rows)
    ]
    
    lat = np.random.uniform(52.36, 52.46, n_rows)
    lon = np.random.uniform(16.85, 17.00, n_rows)
    
    return pd.DataFrame({
        "date": dates,
        "lat": lat,
        "lon": lon
    })

data = generate_data(n_rows=500)

data
```

Funkcja `generate_data()` jest przykładem funkcji, której wyniki mogłyby być zapisane w pamięci podręcznej aby przyspieszyć wykonywanie skryptu. Streamlit wykonuje cały skrypt przy zajściu jakiejkolwiek zmiany, więc przyspieszenie wykonywania takich fragmentów skryptu ma niebagatelne znaczenie. 

- zmień liczbę generowanych punktów na: 200, 1000, 10000 i zaobserwuj czas potrzebny na załadowanie strony
- udekoruj funkcję dekoratorem `@st.cache_data` i porównaj czas ładowania strony

W tej chwili liczba generowanych punktów danych jest ustawiona "na sztywno" na 500. Dodajmy pole tekstowe umożliwiające użytkownikowi podanie tej liczby samodzielnie. Zmodyfikuj powyższy kod tak, żeby możliwe było przekazanie do skryptu pożądanej liczby punktów. Wykorzystaj w tym celu komponent [number_input](https://docs.streamlit.io/library/api-reference/widgets/st.number_input)

```python
@st.cache_data
def generate_data(...):
    ...
    
n_rows = st.number_input(
    label="How many points to generate?",
    min_value=1,
    max_value=100000,
    value=1000
)

data = generate_data(n_rows=n_rows)
```

Dodajmy jeszcze opcję wyświetlenia obok mapy surowych danych


```python
if st.checkbox("Show raw data?"):
    st.subheader("Raw data")
    st.dataframe(data)
```

Streamlit pozwala na dołączanie praktycznie wszystkich rodzajów wykresów do aplikacji. Obsługuje wykresy generowane przez Matplotlib, Altaira, Bokeh, GraphViz i wiele innych. W następnym kroku wyznaczymy histogramy wołań samochodów z dokładnością do godziny i je wyświetlimy


```python
hist_vals = np.histogram(data.date.dt.hour, bins=24)[0]

st.bar_chart(hist_vals)
```

Wygodnym sposobem wygenerowania filtru dla danych numerycznych jest komponent `st.slider`. Pozwolimy teraz użytkownikom zawęzić zakres wyświetlanych godzin.


```python
hour_filter = st.slider("hour", 0, 23, 17)

df_filtered = data[data.date.dt.hour == hour_filter]

st.subheader(f"Map of all uber pickups at {hour_filter}:00")

st.map(df_filtered)
```

W tej chwili komponent `st.slider` pozwala na wybór tylko jednej wartości. Jeśli chcemy wykorzystać przedział godzin do wygenerowania mapy, możemy wykorzystać w tym celu komponent `st.select_slider` ([link do API](https://docs.streamlit.io/library/api-reference/widgets/st.select_slider)).


```python
min_hour, max_hour = st.select_slider(
    label="hours", 
    options=range(24),
    value=(7,16)

filter_idx = (data.date.dt.hour >= min_hour) & (data.date.dt.hour <= max_hour)
df_filtered = data[filter_idx]

st.subheader(f"Map of all uber pickups between {min_hour}:00 and {max_hour}:00")

st.map(df_filtered)
```

Dotychczas wszystkie komponenty reagowały natychmiast na zmiany, co uniemożliwiało jednoczesne zgłoszenie wielu zmiennych. Rozwiązaniem jest wykorzystanie komponentu `st.form` ([link do API](https://docs.streamlit.io/library/api-reference/control-flow/st.form))


```python
with st.form(key="my_form"):
    name_input = st.text_input("Name:")
    dob_input = st.date_input("Date of birth:")
    weight_input = st.number_input("Weight (kg):")
    height_input = st.number_input("Height (cm):")
    
    submit_btn = st.form_submit_button("Compute BMI")
    
if submit_btn:
    bmi = weight_input / (height_input/100)**2
    st.write(f"Hello {name_input}, your BMI={bmi:.2%f})
```

Już wcześniej spotkaliśmy się z koniecznością wykorzystania _placeholdera_ do zapewnienia sobie miejsca na wstawienie danych do wcześniejszego miejsca w skrypcie aplikacji. Poniższy przykład pokazuje sposób obsługi takich komponentów.


```python
st.text("First line")

empty_text = st.empty()
empty_chart = st.empty()

st.text("Fourth line")

empty_text.text("Second line")

empty_chart.line_chart(np.random.randn(50,2))
```

W nagrodę za osiągnięcie tego punktu w tutorialu dodaj na końcu skryptu poniższą komendę:

```python
st.balloons()
```

# Zadanie samodzielne

Załaduj z biblioteki `scikit-learn` zbioór danych o [mieszkaniach w Kaliforni](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html). Następnie przygotuj analizę, na którą będą się składać:

- wyświetlenie zbioru danych
- możliwość wyświetlenia tylko tych nieruchomości, które są młodsze niż 10 lat (checkbox)
- filtr pozwalający na wskazanie zakresu liczby mieszkańców bloku
- wykres pokazujący rozkład średnich wartości domów
- prosty model regresji (np. drzewo decyzyjne) który wyznacza wartość domu na podstawie wybranego podzbioru 3 parametrów
- mapa wizualizująca położenie nieruchomości
- formularz przyjmujący wybrane parametry i wyświetlający predykcję wartości domu

---

# Tworzenie własnych komponentów (zaawansowane)

Streamlit jest bardzo elastycznym środowiskiem, w którym stosunkowo łatwo możemy stworzyć swój własny komponent. Będzie to wymagało jednak oprogramowania zarówno front-endu, jak i back-endu.

W pierwszej kolejności sklonuj lokalnie repozytorium [https://github.com/streamlit/component-template](https://github.com/streamlit/component-template) i zainstaluj potrzebne pakiety npm oraz uruchom serwer webpack.

```bash
bash$ git clone https://github.com/streamlit/component-template

bash$ cd component-template/template/my_component/frontend

bash$ npm install

bash$ npm run start
```

Przejdź do katalogu `component-template/template/my_component` i zmodyfikuj plik `__init__.py`. Umieść w nim poniższy kod


```python
import os
import streamlit as st
import streamlit.components.v1 as components

st.title("My component example")

_component_func = components.declare_component(
    "my_component",
    url="http://localhost:3001",
)

def my_component(start, key=None):
    component_value = _component_func(start=start, key=key, default=100)

    return component_value

counter = my_component(10)

st.markdown(f"You have {counter} clicks left!")
```

Otwórz nowy terminal i uruchom aplikację komponentu

```bash
bash$ streamlit run component-template/my_component/__init__.py
```

Następnie przejdź do katalogu `component-template/template/my_component/fronend/src/MyComponent.tsx` i umieść poniższy kod:


```python
import {
  Streamlit,
  StreamlitComponentBase,
  withStreamlitConnection,
} from "streamlit-component-lib"
import React, { ReactNode } from "react"

interface State {
  counter: number
}

class MyComponent extends StreamlitComponentBase<State> {
  public state = { counter: this.props.args["start"] }

  public render = (): ReactNode => {

    return (
      <span>
        Clicks remaining: {this.state.counter} &nbsp;
        <button
          onClick={this.onClicked}
        >
          Click me!
        </button>
      </span>
    )
  }

  private onClicked = (): void => {
    if (this.state.counter > 0) {
        this.setState(
        prevState => ({ counter: prevState.counter - 1 }),
        () => Streamlit.setComponentValue(this.state.counter)
        )
    } 
  }
}

export default withStreamlitConnection(MyComponent)
```

