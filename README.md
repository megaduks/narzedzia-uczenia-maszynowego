# narzedzia-uczenia-maszynowego

Repozytorium zawiera materiały dydaktyczne na potrzeby przedmiotu "Narzędzia uczenia maszynowego" realizowanego na specjalności "sztuczna inteligencja" na II stopniu studiów na kierunku Informatyka na Politechnice Poznańskiej

Do realizacji ćwiczeń służy maszyna wirtualna `Ubuntu-22.04-NUM` umieszczona w środowisku VLab. Po uruchomieniu maszyny wirtualnej proszę się zalogować na konto `student` i utworzyć obraz kontenera zgodnie z poleceniami dla danego ćwiczenia.

---

### DVC

Przejdź do katalogu `lab-01-dvc`, zbuduj kontener Docker, uruchom go i kontynuuj zgodnie z instrukcjami zamieszczonymi w `lab-01-dvc/README.md`.

```bash
cd lab-01-dvc
sudo docker build -t dvc:latest .
sudo docker container run -it dvc:latest /bin/bash
```

### Snorkel

Przejdź do katalogu `lab-02-snorkel`, zbuduj kontener Docker i uruchom go (instrukcja poniżej). Po uruchomieniu kontenera zobaczysz adresy, pod którymi serwer `jupyter` jest dostępny poza kontenerem. Otwórz jeden z adresów, uruchom w przeglądarce plik `snorkel.ipynb` i wykonaj ćwiczenie.

```bash
cd lab-02-snorkel
sudo docker build -t snorkel:latest .
sudo docker container run -it -p 8888:8888 snorkel:latest
```

### Streamlit

Przejdź do katalogu `lab-03-streamlit`, zbuduj kontener Docker i uruchom go (instrukcja poniżej). Po uruchomieniu kontenera zobaczysz adres, pod którym działa aplikacja Streamlit.

```bash
cd lab-03-streamlit
sudo docker build -t streamlit:latest .
sudo docker container run -it -p 8501:8501 streamlit:latest
```

Otwórz nowe okno konsoli i sprawdź identyfikator uruchomionego kontenera Docker. Korzystając z tego identyfikatora uruchom konsolę wewnątrz kontenera. Ze względu na ogólną trudność współdzielenia clipboardu między kontenerem i hostem, najprościej jest wykonać ćwiczenie uruchamiając w konsoli edytor `vim` i dzieląc ekran na dwie części (komenda `:split`). Przechodzenie między podzielonymi panelami w `vim` jest realizowane przez sekwencję klawiszy `ctrl-W ctrl-W`.

```bash
sudo docker ps
sudo docker exec -it <container-id> /bin/bash
vim -o streamlit.md helloworld.py 
```

### Ludwig

Przejdź do katalogu `lab-04-ludwig`, zbuduj kontener Docker i uruchom go (instrukcja poniżej). Po uruchomieniu kontenera wejdź do linii poleceń i wykonaj instrukcje zawarte w pliku `ludwig.md`

```bash
cd lab-04-ludwig
sudo docker build -t ludwig:latest .
sudo docker container run -it -p 8081:8081 ludwig:latest /bin/bash
```

### Prodigy

Przejdź do katalogu `lab-05-prodigy`, zbuduj kontener Docker i uruchom go (instrukcja poniżej). Po uruchomieniu kontenera wejdź do linii poleceń i wykonaj instrukcje zawarte w pliku `prodigy.md`

```bash
cd lab-05-prodigy
docker build -t prodigy:latest .
docker container run -it -p 8080:8080 prodigy:latest /bin/bash
```

### MLFlow

Przejdź do katalogu `lab-06-mlflow`, zbuduj kontener Docker i uruchom go (instrukcja poniżej). Po uruchomieniu kontenera wejdź do linii poleceń i wykonaj instrukcje zawarte w pliku `mlflow.md`

```bash
cd lab-06-mlflow
docker build -t mlflow:latest .
docker container run -it -p 5000:5000 mlflow:latest /bin/bash
```

### nlpaug & checklist

Przejdź do katalogu `lab-07-nlpaug`, zbuduj kontener Docker i uruchom go (instrukcja poniżej). Po uruchomieniu kontenera wejdź do uruchomionego notatnika i wykonaj ćwiczenie.

```bash
cd lab-07-mlflow
docker build -t nlpaug:latest .
docker container run -it -p 8888:8888 nlpaug:latest
```
