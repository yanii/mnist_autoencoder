#!/usr/bin/env bash
./test_mnist.py --gpu 0 --model mlp.model --net bn 1>&2 | tee -a test.log
