FROM nvidia/cuda:11.4.0-cudnn8-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
COPY . /app

RUN apt-get update
RUN apt-get install -yq --no-install-recommends python3-pip python3-tk  python3.9 python3.9-dev python3.9-distutils ffmpeg libsm6 libxext6
RUN python3.9 -m pip install -r /app/requirements.txt

RUN rm -rf /var/lib/apt/lists/*
ENV PYTHONPATH=/app

CMD ["/bin/bash"]