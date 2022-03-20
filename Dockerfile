ARG BASE_IMAGE
FROM $BASE_IMAGE

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
RUN python -m pip install --upgrade pip
RUN python -m pip install --no-cache-dir --user opencv-python-headless==4.5.5.64

# Install PyTorch on CPU
RUN nvidia-smi; if [ "$?" != 0 ]; then \
  echo "Installing PyTorch on CPU"; \
  python -m pip install --no-cache-dir --user torch==1.11.0+cpu -f https://download.pytorch.org/whl/torch_stable.html; \
  fi

ENTRYPOINT ["python", "colorize.py"]
