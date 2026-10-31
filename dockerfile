FROM quay.io/jupyter/scipy-notebook:latest

WORKDIR /home/jovyan/work

COPY requirements.txt .
RUN pip install -r requirements.txt

#to allow user read and write access
COPY --chown=${NB_UID}:${NB_GID} data/dataset.csv data/dataset.csv
COPY --chown=${NB_UID}:${NB_GID} notebook notebook