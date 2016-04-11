#/bin/bash
./train_mnist.py --net bn --gpu 0 --epoch 5000 1>&2 |tee -a train.log
