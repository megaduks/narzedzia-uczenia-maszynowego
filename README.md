# narzedzia-uczenia-maszynowego

Repozytorium zawiera materiały dydaktyczne na potrzeby przedmiotu "Narzędzia uczenia maszynowego" realizowanego na specjalności "sztuczna inteligencja" na II stopniu studiów na kierunku Informatyka na Politechnice Poznańskiej

---

### DVC

Przejdź do katalogu `lab-01-dvc`, zbuduj kontener Docker, uruchom go i kontynuuj zgodnie z instrukcjami zamieszczonymi w `lab-01-dvc/README.md`.

```bash
cd lab-01-dvc
docker build -t dvc:latest .
docker container run -it dvc:latest /bin/bash
```
