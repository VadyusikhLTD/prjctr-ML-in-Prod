FROM python:3.8-slim-buster

LABEL mainteiner="Vadym Honcharenko"

WORKDIR /src

COPY src/requirements.txt requirements.txt
RUN python3 -m pip install -r requirements.txt

COPY src .

CMD python ./main.py