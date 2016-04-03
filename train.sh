#/usr/bin/env bash
./train_mnist.py --net bn --gpu 0 --epoch 1000 2>&1 | tee -a  train.log
