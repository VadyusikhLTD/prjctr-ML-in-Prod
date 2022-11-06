FROM python:3.9-slim-buster

WORKDIR /app

RUN apk add --update nano

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

RUN ln -s /usr/bin/python3 /usr/bin/python

ENV PYTHONPATH /app
COPY . . 

CMD [ "bash" ]
