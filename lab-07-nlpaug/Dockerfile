FROM python:3.9

WORKDIR /home

COPY . /home

RUN pip install notebook jupyterlab

RUN apt-get update \
	&& curl -sL https://deb.nodesource.com/setup_19.x | bash - \
	&& apt-get install -y nodejs vim 

COPY requirements.txt /home/requirements.txt
RUN pip install -r /home/requirements.txt

WORKDIR /home

RUN wget https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip
RUN unzip /home/wiki-news-300d-1M.vec.zip

RUN spacy download en_core_web_sm

RUN jupyter nbextension install --py --sys-prefix checklist.viewer
RUN jupyter nbextension enable checklist.viewer --py --sys-prefix
RUN jupyter nbextension enable --py --sys-prefix checklist.viewer

EXPOSE 8888
ENTRYPOINT ["jupyter", "notebook","--ip=0.0.0.0","--allow-root"]
