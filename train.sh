#/bin/bash
./train_mnist.py --net bn --gpu 1 --epoch 200 1>&2 |tee -a train.log
