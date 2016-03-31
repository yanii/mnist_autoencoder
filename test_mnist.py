#!/usr/bin/env python
"""Chainer example: train a multi-layer perceptron on MNIST

This is a minimal example to write a feed-forward net.

"""
from __future__ import print_function
import argparse
import time

import numpy as np
import six
import math
import scipy.misc
import matplotlib.pyplot as plt

import chainer
from chainer import computational_graph
from chainer import cuda
import chainer.links as L
from chainer import optimizers
from chainer import serializers

import data
from autoencoder import AutoEncoder,CrossEntropyAutoEncoder

parser = argparse.ArgumentParser(description='Chainer example: MNIST')
parser.add_argument('--model', '-m',
                    help='Initialize the model from given file')
parser.add_argument('--net', '-n', choices=('normal', 'bn'),
                    default='normal', help='Network type')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--batchsize', '-b', type=int, default=100,
                    help='learning minibatch size')
args = parser.parse_args()

batchsize = args.batchsize

print('GPU: {}'.format(args.gpu))
print('# Minibatch-size: {}'.format(args.batchsize))
print('Network type: {}'.format(args.net))
print('')

# Prepare dataset
print('load MNIST dataset')
mnist = data.load_mnist_data()
mnist['data'] = mnist['data'].astype(np.float32)
mnist['data'] /= 255
mnist['target'] = mnist['target'].astype(np.int32)

N = 60000
x_train, x_test = np.split(mnist['data'],   [N])
y_train, y_test = np.split(mnist['target'], [N])
N_test = y_test.size

INPUT_SIZE  = 28 * 28 # 784
OUTPUT_SIZE = 30
LAYER_SIZES = [INPUT_SIZE, 1000, 500, 250, OUTPUT_SIZE]
SAVE_IMAGES=True

# Prepare multi-layer perceptron model, defined in net.py
if args.net == 'normal':
    ae = AutoEncoder(LAYER_SIZES, use_bn=False)
    aeback = AutoEncoder(LAYER_SIZES, use_bn=False, forwardchain=ae)
else:
    ae = AutoEncoder(LAYER_SIZES, use_bn=True)
    aeback = AutoEncoder(LAYER_SIZES, use_bn=True, forwardchain=ae)

model = CrossEntropyAutoEncoder(ae, aeback)

if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    model.to_gpu()
xp = np if args.gpu < 0 else cuda.cupy

# Init/Resume
print('Load model from', args.model)
serializers.load_npz(args.model, model)

# evaluation
sum_mean_squared_error = 0
sum_loss = 0

#if SAVE_IMAGES:
#    plt.axis('off')
#    plt.ion()
#    plt.show()

for i in six.moves.range(0, N_test, batchsize):
    x = chainer.Variable(xp.asarray(x_test[i:i + batchsize]),
                         volatile='on')
    t = chainer.Variable(xp.asarray(y_test[i:i + batchsize]),
                         volatile='on')
    loss = model(x, t)
    sum_loss += float(loss.data) * len(t.data)
    sum_mean_squared_error += float(model.mean_squared_error.data) * len(t.data)

    if SAVE_IMAGES:
        y = aeback(ae(x))

        if args.gpu >= 0:
            x_cpu = cuda.to_cpu(x.data)
            y_cpu = cuda.to_cpu(y.data)
        else:
            x_cpu = x.data
            y_cpu = y.data

        imagesize = math.sqrt(x_cpu[0].shape[0])
        for j in six.moves.range(0, batchsize):
            stack = np.hstack(
                (
                    np.reshape(x_cpu[j], (imagesize, imagesize)),
                    np.reshape(y_cpu[j], (imagesize, imagesize))
                    )
                )

            stack = stack*255
            image = scipy.misc.toimage(stack, cmin=0.0, cmax=255)
            image.save('images/'+str(j+i)+'.png')
            #plt.imshow(image, cmap='gist_gray', interpolation='none', vmin=0, vmax=255)
            #plt.draw()
            #plt.pause(0.001)

print('test  mean loss={}, MSE={}'.format(
    sum_loss / N_test, sum_mean_squared_error / N_test))
