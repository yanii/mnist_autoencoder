#/usr/bin/env python
./train_mnist.py --net bn --gpu 0 --epoch 1000 1>&2| tee -a  train.log
