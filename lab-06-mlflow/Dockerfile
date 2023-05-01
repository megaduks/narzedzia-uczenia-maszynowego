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

COPY train.py /home/
COPY mlflow.md /home/
COPY data /home/data/

EXPOSE 5000
