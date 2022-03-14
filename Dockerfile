FROM python:3.9-slim-buster

ENV DEBIAN_FRONTEND noninteractive

# Setup user
ARG USER=colorizer
RUN useradd -ms /bin/bash $USER
RUN echo "root:Docker!" | chpasswd
ENV HOME=/home/$USER
ENV PATH=$HOME/.local/bin:$PATH
ENV shell=/bin/bash

# Install system dependencies
USER root
RUN apt-get update && apt-get install -y curl wget

# Install Python dependencies
USER $USER
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip
RUN python -m pip install --no-cache-dir --user -r requirements.txt

ENTRYPOINT ["python", "colorize.py"]
