FROM ludwigai/ludwig

WORKDIR /home

RUN pip3 install ludwig[vis,text,images]

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    vim \
    && rm -rf /var/lib/apt/lists/*

COPY data /home/data
COPY ludwig.md /home/ludwig.md

