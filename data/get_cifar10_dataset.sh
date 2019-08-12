#!/bin/bash

CURRENT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo ${CURRENT_DIR}
cd ${CURRENT_DIR}

FILE=cifar-10-python.tar.gz
DIR=cifar-10-batches-py
URL=http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz

if [ -f ${FILE} ] || [ -d ${DIR} ]; then
    echo "file exists"
else
    echo "* downloading dataset ..."
    wget ${URL}
fi

if [ -f ${FILE} ] && ! [ -d "cifar10" ]; then
    mkdir cifar10
    cd cifar10
    echo "* unzipping..."
    tar -xvzf ../${FILE}
    rm ../${FILE}
fi
