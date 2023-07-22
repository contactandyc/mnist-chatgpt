#!/bin/bash

download_file () {
  curl -s $1 -o /dev/stdout | gunzip -c > $2
}

download_file http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz train-images-idx3-ubyte
download_file http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz train-labels-idx1-ubyte
download_file http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz t10k-images-idx3-ubyte
download_file http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz t10k-labels-idx1-ubyte
