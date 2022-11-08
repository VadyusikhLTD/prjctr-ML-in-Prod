FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime as base

RUN apt-get update && apt-get install make

WORKDIR /app

RUN pip install pip --upgrade
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

RUN ln -s /usr/bin/python3 /usr/bin/python

ENV PYTHONPATH /app:/app/src:/app/src/utils
COPY . .

CMD [ "bash" ]

# Streamlit UI docker image
FROM base AS app-streamlit
CMD streamlit run --server.address 0.0.0.0 --server.port 8080 src/steamlit_serving.py

# Fast API docker image
FROM base AS app-fastapi
CMD uvicorn --host 0.0.0.0 --port 8081 --workers 2  src.fast_api_serving:app