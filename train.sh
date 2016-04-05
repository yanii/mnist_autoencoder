#/bin/bash
./train_mnist.py --net bn --gpu 0 --epoch 500 1>&2 |tee -a train.log
