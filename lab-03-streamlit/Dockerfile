FROM python:3.9-slim

WORKDIR /home

COPY requirements.txt /home/
RUN pip3 install -r requirements.txt

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    vim \
    && rm -rf /var/lib/apt/lists/*

COPY helloworld.py /home/helloworld.py
COPY titanic.csv /home/titanic.csv
COPY streamlit.md /home/streamlit.md

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "helloworld.py", "--server.port=8501", "--server.address=0.0.0.0"]
