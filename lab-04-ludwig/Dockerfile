FROM python:3.8-slim

WORKDIR /home

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    vim \
    tmux \
    && rm -rf /var/lib/apt/lists/*

RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
RUN . $HOME/.cargo/env

RUN pip install tokenizers
RUN pip install ludwig
RUN pip install ludwig[serve]

COPY data /home/data
COPY image-test.csv /home/image-test.csv
COPY image-train.csv /home/image-train.csv
COPY ludwig.md /home/ludwig.md

EXPOSE 8081

