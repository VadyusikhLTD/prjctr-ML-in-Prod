FROM python:3.8-slim-buster

LABEL mainteiner="Vadym Honcharenko"

RUN python3 -m pip install requests
#RUN python -m pip install -q 'requests==2.28.1' 'json==2.0.9'
RUN mkdir "src"

COPY ./src/. src/.

CMD python ./src/main.py