#!/bin/bash

CURRENT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo ${CURRENT_DIR}
cd ${CURRENT_DIR}

FILE=dtd-r1.0.1.tar.gz
DIR=dtd
URL=https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz

if [ -f ${FILE} ] || [ -d ${DIR} ]; then
    echo "file exists"
else
    echo "* downloading dataset ..."
    wget ${URL} -O ${FILE}
fi

if [ -f ${FILE} ] && ! [ -d ${DIR} ]; then
	echo "* unzipping..."
  tar -xvzf ${FILE}

	echo "* change folder to unzipped dataset"
	cd dtd

	echo "* create 'train' and 'test' folders"
	mkdir train test
	echo "* inside train and test create folders for each class"
	ls images/ | sed 's;^;mkdir train/;' | source /dev/stdin
	ls images/ | sed 's;^;mkdir test/;' | source /dev/stdin
	echo "* copy images from the 'train1.txt' and 'val1.txt' lists into 'train' folder"
	cat labels/train1.txt | awk -F'/' '{print "mv images/"$1 "/" $2 " train/" $1}' | source /dev/stdin
	cat labels/val1.txt | awk -F'/' '{print "mv images/"$1 "/" $2 " train/" $1}' | source /dev/stdin
	echo "* copy images from 'test1.txt' into 'test' folder"
	cat labels/test1.txt | awk -F'/' '{print "mv images/"$1 "/" $2 " test/" $1}' | source /dev/stdin
	rm -fr images
  rm ../${FILE}
	echo "DONE"
fi
