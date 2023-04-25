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

COPY prodigy-1.11.10-cp39-cp39-linux_x86_64.whl /home/
COPY homebrewing.jsonl.bz2 /home/
COPY programming.jsonl.bz2 /home/
COPY prodigy.json /home/
COPY prodigy.md /home/
COPY recipe.py /home/
COPY images /home/images/

RUN python -m spacy download en_core_web_lg
RUN pip install prodigy -f prodigy-1.11.10-cp39-cp39-linux_x86_64.whl

EXPOSE 8080
