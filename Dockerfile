FROM nvidia/cuda:10.2-runtime-ubuntu18.04
COPY . /usr/text_similarity/
WORKDIR /usr/text_similarity/
RUN apt update && apt install -y --no-install-recommends \
    git \
    build-essential \
    python3-dev \
    python3-pip \
    python3-setuptools
RUN pip3 -q install pip --upgrade
RUN pip3 install -U pip setuptools wheel
RUN pip3 install -r requirements.txt
ENV PYTHONPATH "${PYTHONPATH}:/usr/text_similarity/src"
