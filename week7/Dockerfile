FROM python:3.9-buster as base_local

WORKDIR /app

RUN apt-get update
RUN apt-get install nano

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

#RUN #ln -s /usr/bin/python3 /usr/bin/python


FROM fs:base as fs
ENV PYTHONPATH /app
RUN ls
RUN rm *
RUN ls
COPY . .
RUN ls

CMD [ "bash" ]
