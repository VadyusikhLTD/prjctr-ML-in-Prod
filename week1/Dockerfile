FROM python:3.8-slim-buster

LABEL mainteiner="Vadym Honcharenko"

COPY src src/.
WORKDIR /src

RUN python3 -m  pip install -r requirements.txt

CMD python ./main.py