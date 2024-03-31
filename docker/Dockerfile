from nvidia/cuda:11.7.1-devel-ubuntu20.04
RUN su -
RUN apt-get update
RUN apt-get -y install sudo
#python install
RUN apt install build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev libsqlite3-dev wget libbz2-dev liblzma-dev -y
RUN wget https://www.python.org/ftp/python/3.11.3/Python-3.11.3.tgz && tar -xf Python-3.11.3.tgz && cd Python-3.11.3 && ./configure --enable-optimizations && echo Y | make -j 12 && sudo make altinstall
RUN apt-get install git -y


RUN pip3.11 install --upgrade pip
# && pip3.11 install flash-attn --no-build-isolation
RUN apt-get install vim -y