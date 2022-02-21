# syntax=docker/dockerfile:1

FROM python:3.8-slim-buster

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

RUN mkdir /pre
WORKDIR /pre
COPY prediction.py /pre

ENTRYPOINT ["flask", "run", "--port", "5010", "--host", "0.0.0.0"]