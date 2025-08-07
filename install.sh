#!/bin/sh

[ $(which python3-pip) ] || sudo apt install python3-pip

mkdir runs/

pip install -r ./requirements.txt