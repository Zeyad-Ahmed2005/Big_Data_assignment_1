FROM quay.io/jupyter/scipy-notebook:latest

WORKDIR /home/jovyan/work

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY data data
COPY notebook notebook