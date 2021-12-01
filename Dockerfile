FROM python:3.9-slim-buster

RUN apt-get update && apt-get install -y curl ffmpeg libsm6 libxext6

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt
