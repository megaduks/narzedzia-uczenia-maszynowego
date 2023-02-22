FROM python

RUN apt update && apt install vim tree git python3-dev gcc libc-dev libffi-dev -y

COPY data /home/data
COPY src /home/src
COPY requirements.txt /home/

WORKDIR /home
RUN pip install -r /home/requirements.txt
