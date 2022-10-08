FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime
WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

RUN ln -s /usr/bin/python3 /usr/bin/python

ENV PYTHONPATH /app/image_classification
COPY . .

RUN apt-get update && apt-get install make

CMD [ "bash" ]