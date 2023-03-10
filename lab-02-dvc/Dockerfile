FROM python

WORKDIR /home

COPY requirements.txt /home/
RUN pip install -r /home/requirements.txt

RUN apt update && apt install vim tree git python3-dev gcc libc-dev -y

COPY data /home/data
COPY snorkel.ipynb /home/

RUN python -m spacy download en_core_web_sm

ENTRYPOINT ["jupyter", "notebook", "--ip=0.0.0.0", "--no-browser", "--allow-root"]
