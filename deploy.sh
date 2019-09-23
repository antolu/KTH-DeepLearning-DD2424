#!/bin/sh

if [ ! -d $PWD/docs ]; then
    echo "You probably need to cd to the repository root directory"
    exit 1
fi

mkdir $PWD/datasets
cd $PWD/datasets
echo "Downloading CIFAR-10 matlab version..."
curl -OL https://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz

echo "Extracting dataset"

tar -xvf ./cifar-10-matlab.tar.gz
mv ./cifar-10-batches-mat ./cifar-10
rm -f ./cifar-10-matlab.tar.gz
