#!/bin/bash

#sudo apt update
#sudo apt install -y python3-venv libpython3-dev graphviz ninja-build default-jre
#python3 -m venv venv

#conda install -c conda-forge gxx gcc libgcc-ng sysroot_linux-64 cmake elfutils libunwind
#conda update libgcc-ng
#conda install -c anaconda graphviz hdf5

conda install -c nvidia/label/cuda-11.7.0 -c conda-forge cuda=11.7.0 python=3.9.0 gcc=9.4.0 gxx=9.4.0 cmake boost

conda install -c nvidia/label/cuda-12.1.1 -c conda-forge cuda=12.1.1 python=3.10.0 gcc=11.3.0 gxx=11.3.0 cmake boost

# conda install -c conda-forge graphviz python=3.9.0 gcc=9.4.0 gxx=9.4.0 boost
python -m venv venv
venv/bin/pip install torch==1.11.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
venv/bin/pip install torch==2.0.1+cu118 -f https://download.pytorch.org/whl/torch_stable.html
venv/bin/pip install -r requirements.txt
