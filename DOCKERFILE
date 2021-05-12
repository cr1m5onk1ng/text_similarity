FROM nvidia/cuda:10.2-runtime
COPY . /usr/text_similarity/
WORKDIR /usr/text_similarity/
RUN apt update && apt install -y --no-install-recommends \
    git \
    build-essential \
    python3-dev \
    python3-pip \
    python3-setuptools
RUN pip install --upgrade pip
RUN pip install -U pip setuptools wheel
RUN pip install -r requirements.txt
ENV PYTHONPATH "${PYTHONPATH}:/usr/text_similarity/src"
