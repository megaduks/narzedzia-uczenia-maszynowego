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
	wget \
    openssl \
    zlib1g-dev \
    libssl-dev \
    libffi-dev \
    libbz2-dev \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh \
    && echo "Running $(conda --version)" && \
    conda init bash && \
    . /root/.bashrc && \
    conda update conda && \
    conda create -n python-app && \
    conda activate python-app && \
    conda install python=3.6 pip 

COPY train.py /home/
COPY mlflow.md /home/
COPY regression /home/regression/
COPY wine-quality /home/wine-quality/
COPY data /home/data/

EXPOSE 5000
