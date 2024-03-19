# base image
FROM tensorflow/tensorflow:latest-gpu-jupyter

LABEL maintainer "Natanael Moura Junior <natmourajr@lps.ufrj.br>"

USER root
ENV LC_ALL C.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV TERM screen

RUN ls

RUN ln -snf /usr/share/zoneinfo/Etc/UTC /etc/localtime \
    && echo "Etc/UTC" > /etc/timezone

RUN \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    texlive-latex-base \
    texlive-latex-extra \
    texlive-science \
    texlive-fonts-recommended \
    xzdec \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    git \
    vim \ 
    less \
    nano \
    nodejs \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


# Set python 3 as the default python
# RUN update-alternatives --set python /usr/local/bin/python3
RUN apt-get update -y
RUN apt-get install -y libsndfile-dev
RUN pip install --no-cache-dir --upgrade pip setuptools wheel isort
RUN pip install jax[cuda11_cudnn82] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Install pip packages
# RUN mkdir /images
# RUN cd /images && git clone https://github.com/natmourajr/docker.git && cd docker/CompactacaoDeSinais && pip install -r requirements.txt
COPY requirements.txt ./
RUN pip install -r requirements.txt
# RUN rm -rf /images
RUN cd /
RUN rm -rf tensorflow-tutorials

