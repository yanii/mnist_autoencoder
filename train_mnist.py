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
from autoencoder import AutoEncoder,CrossEntropyAutoEncoder,MSEAutoEncoder

parser = argparse.ArgumentParser(description='Chainer example: MNIST')
parser.add_argument('--initmodel', '-m', default='',
                    help='Initialize the model from given file')
parser.add_argument('--resume', '-r', default='',
                    help='Resume the optimization from snapshot')
parser.add_argument('--net', '-n', choices=('normal', 'bn'),
                    default='normal', help='Network type')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--epoch', '-e', default=20, type=int,
                    help='number of epochs to learn')
parser.add_argument('--batchsize', '-b', type=int, default=100,
                    help='learning minibatch size')
args = parser.parse_args()

batchsize = args.batchsize
n_epoch = args.epoch

print('GPU: {}'.format(args.gpu))
print('# Minibatch-size: {}'.format(args.batchsize))
print('# epoch: {}'.format(args.epoch))
print('Network type: {}'.format(args.net))
print('')

# Prepare dataset
print('load MNIST dataset')
mnist = data.load_mnist_data()
mnist['data'] = mnist['data'].astype(np.float32)
mnist['data'] /= 255
mnist['target'] = mnist['target'].astype(np.int32)

#N_test = 10000
N_val = 10000
N = 60000 - N_val
x_train, x_val, x_test = np.split(mnist['data'],   [N, N+N_val])
y_train, y_val, y_test = np.split(mnist['target'], [N, N+N_val])
N_test = y_test.size
print ('train size:', y_train.size, 'val size: ', y_val.size, 'test size:', y_test.size)

WEIGHT_DECAY = 0.00005
INIT_LR = 0.5

INPUT_SIZE  = 28 * 28 # 784
OUTPUT_SIZE = 30
LAYER_SIZES = [INPUT_SIZE, 1000, 500, 250, OUTPUT_SIZE]
SAVE_IMAGES=True
N_IMAGES_SAVE=10

# Prepare multi-layer perceptron model, defined in net.py
if args.net == 'normal':
    ae = AutoEncoder(LAYER_SIZES, use_bn=False)
    aeback = AutoEncoder(LAYER_SIZES, use_bn=False, forwardchain=ae)
else:
    ae = AutoEncoder(LAYER_SIZES, use_bn=True)
    aeback = AutoEncoder(LAYER_SIZES, use_bn=True, forwardchain=ae)

#model = CrossEntropyAutoEncoder(ae, aeback)
model = MSEAutoEncoder(ae, aeback)

if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    model.to_gpu()
xp = np if args.gpu < 0 else cuda.cupy

# Setup optimizer
optimizer = optimizers.MomentumSGD()
optimizer.setup(model)
optimizer.lr = INIT_LR
if WEIGHT_DECAY > 0:
    optimizer.add_hook(chainer.optimizer.WeightDecay(WEIGHT_DECAY))

# Init/Resume
if args.initmodel:
    print('Load model from', args.initmodel)
    serializers.load_npz(args.initmodel, model)
if args.resume:
    print('Load optimizer state from', args.resume)
    serializers.load_npz(args.resume, optimizer)

# Learning loop
for epoch in six.moves.range(1, n_epoch + 1):
    print('epoch', epoch, 'lr', optimizer.lr)

    # training
    perm = np.random.permutation(N)
    sum_mean_squared_error = 0
    sum_loss = 0
    start = time.time()
    for i in six.moves.range(0, N, batchsize):
        x = chainer.Variable(xp.asarray(x_train[perm[i:i + batchsize]]))
        t = chainer.Variable(xp.asarray(y_train[perm[i:i + batchsize]]))

        # Pass the loss function (Classifier defines it) and its arguments
        model.setTrain()
        optimizer.update(model, x)
        iterations = 1+(((epoch-1)*N)+i)/batchsize
        optimizer.lr = float(INIT_LR)/(1.0 + float(INIT_LR)*WEIGHT_DECAY*iterations)

        if epoch == 1 and i == 0:
            with open('graph.dot', 'w') as o:
                g = computational_graph.build_computational_graph(
                    (model.loss, ))
                o.write(g.dump())
            print('graph generated')

        model.setTest()
        loss = model(x)
        sum_loss += float(loss.data)# * len(t.data)
        sum_mean_squared_error += float(model.mean_squared_error.data)# * len(t.data)

    end = time.time()
    elapsed_time = end - start
    throughput = N / elapsed_time
    print('train mean loss={}, MSE={}, iterations={}, throughput={} images/sec'.format(
        sum_loss / N, sum_mean_squared_error / N, iterations, throughput))


    # evaluation
    sum_mean_squared_error = 0
    sum_loss = 0

    if SAVE_IMAGES:
        plt.axis('off')
        plt.ion()
        plt.show()

    for i in six.moves.range(0, N_val, batchsize):
        x = chainer.Variable(xp.asarray(x_val[i:i + batchsize]),
                             volatile='on')
        t = chainer.Variable(xp.asarray(y_val[i:i + batchsize]),
                             volatile='on')
        model.setTest()
        loss = model(x)
        sum_loss += float(loss.data)# * len(t.data)
        sum_mean_squared_error += float(model.mean_squared_error.data)# * len(t.data)

        if SAVE_IMAGES and i == 0:
            y = model.y

            if args.gpu >= 0:
                images = cuda.to_cpu(x.data)[0:N_IMAGES_SAVE]
                y_cpu = cuda.to_cpu(y.data)
            else:
                images = x.data[0:N_IMAGES_SAVE]
                y_cpu = y.data

            vstacked = []
            for i in xrange(len(images)):
                imagesize = math.sqrt(images[0].shape[0])
                vstack = np.vstack(
                    (
                        images[i].reshape((imagesize, imagesize)),
                        y_cpu[i].reshape((imagesize, imagesize))
                    )
                )
                vstacked.append(vstack)
            stack = np.hstack(vstacked)

            stack = stack*255
            image = scipy.misc.toimage(stack, cmin=0.0, cmax=255)
            #image.save('images/'+str(i)+'.png')
            plt.imshow(image, cmap='gist_gray', interpolation='none', vmin=0, vmax=255)
            #plt.tight_layout()
            plt.draw()
            plt.pause(0.002)
    print('val  mean loss={}, MSE={}'.format(
        sum_loss / N_val, sum_mean_squared_error / N_val))

    if epoch % 50 == 0:
        # Save the model and the optimizer
        print('save the model')
        serializers.save_npz('autoencoder.model', model)
        print('save the optimizer')
        serializers.save_npz('autoencoder.state', optimizer)

# Save the model and the optimizer
print('save the model')
serializers.save_npz('autoencoder.model', model)
print('save the optimizer')
serializers.save_npz('autoencoder.state', optimizer)
