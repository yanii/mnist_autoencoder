import chainer
import chainer.functions as F
import chainer.links as L

import numpy as np

class CrossEntropyAutoEncoder(chainer.Chain):
    def __init__(self, autoencoder, autoencoderback):
        super(CrossEntropyAutoEncoder, self).__init__(
            autoencoder=autoencoder,
            autoencoderback=autoencoderback
        )
        autoencoderback.train = False

    def __call__(self, x):
        h = self.autoencoder(x)
        self.y = self.autoencoderback(h)
        self.loss = F.cross_entropy(self.y, x)
        self.mean_squared_error = F.mean_squared_error(self.y*255, x*255)
        return self.loss

    def setTrain(self):
        self.autoencoder.train = True
        self.autoencoderback.train = True
    def setTest(self):
        self.autoencoder.train = False
        self.autoencoderback.train = False

class MSEAutoEncoder(chainer.Chain):
    def __init__(self, autoencoder, autoencoderback):
        super(MSEAutoEncoder, self).__init__(
            autoencoder=autoencoder,
            autoencoderback=autoencoderback
        )

    def __call__(self, x):
        h = self.autoencoder(x)
        self.y = self.autoencoderback(h)
        self.loss = F.mean_squared_error(self.y, x)
        self.mean_squared_error = F.mean_squared_error(self.y*255, x*255)
        return self.loss

    def setTrain(self):
        self.autoencoder.train = True
        self.autoencoderback.train = True
    def setTest(self):
        self.autoencoder.train = False
        self.autoencoderback.train = False


class AutoEncoder(chainer.Chain):
    def __init__(self, layer_sizes, forwardchain=None, use_bn=True, nobias=True,
                 activation_type = F.relu, train=True):
        self.use_bn = use_bn
        self.nobias = nobias
        self.activation = activation_type
        self.forwardchain = forwardchain
        self.train = train

        assert(len(layer_sizes) == 5)
        if self.forwardchain:
            self.layer_sizes = layer_sizes.reverse()
        else:
            self.layer_sizes = layer_sizes

        # Create and register three layers for this MLP
        super(AutoEncoder, self).__init__(
            layer1 = L.Linear(layer_sizes[0], layer_sizes[1], nobias=self.nobias,
            initialW=np.random.normal(0,
                np.sqrt( (2. if self.activation == F.relu else 1.) / (layer_sizes[0]*layer_sizes[1])),
                (layer_sizes[1], layer_sizes[0]))),
            norm1 = L.BatchNormalization(layer_sizes[1]),

            layer2 = L.Linear(layer_sizes[1], layer_sizes[2], nobias=self.nobias,
                initialW=np.random.normal(0,
                    np.sqrt( (2. if self.activation == F.relu else 1.) / (layer_sizes[1]*layer_sizes[2])),
                    (layer_sizes[2], layer_sizes[1]))),
            norm2 = L.BatchNormalization(layer_sizes[2]),

            layer3 = L.Linear(layer_sizes[2], layer_sizes[3], nobias=self.nobias,
                initialW=np.random.normal(0,
                    np.sqrt( (2. if self.activation == F.relu else 1.) / (layer_sizes[2]*layer_sizes[3])),
                    (layer_sizes[3], layer_sizes[2]))),
            norm3 = L.BatchNormalization(layer_sizes[3]),

            layer4 = L.Linear(layer_sizes[3], layer_sizes[4], nobias=self.nobias,
                initialW=np.random.normal(0,
                    np.sqrt( (2. if self.activation == F.relu else 1.) / (layer_sizes[3]*layer_sizes[4])),
                    (layer_sizes[4], layer_sizes[3]))),
        )
        self.linear_layers = [self.layer1, self.layer2, self.layer3, self.layer4]
        self.norm_layers = [self.norm1, self.norm2, self.norm3]

        if self.forwardchain:
            assert(isinstance(forwardchain, AutoEncoder))
            for index in range(len(self.linear_layers)-1):
                src = self.forwardchain.linear_layers[len(self.linear_layers) - 1 - index].__dict__
                dst = self.linear_layers[index].__dict__
                # Note: We do a shallow copy on purpose! Want weight sharing
                #dst = src
                for name in self._params:
                    dst[name] = src[name].T

            for index in range(len(self.norm_layers)-1):
                src = self.forwardchain.norm_layers[len(self.norm_layers) - 1 - index].__dict__
                dst = self.norm_layers[index].__dict__
                # Note: We do a shallow copy on purpose! Want weight sharing
                #dst = src
                for name in self._params:
                    dst[name] = src[name].T

    def __call__(self, x):
        # Forward propagation
        if self.use_bn:
            #h1 = self.activation(F.dropout(self.norm1(self.layer1(x), test=not self.train), ratio=0.1, train=self.train))
            #h2 = self.activation(F.dropout(self.norm2(self.layer2(h1), test=not self.train), ratio=0.1, train=self.train))
            #h3 = self.activation(F.dropout(self.norm3(self.layer3(h2), test=not self.train), ratio=0.1, train=self.train))
            h1 = self.activation(self.norm1(self.layer1(x), test=not self.train))
            h2 = self.activation(self.norm2(self.layer2(h1), test=not self.train))
            h3 = self.activation(self.norm3(self.layer3(h2), test=not self.train))
        else:
            h1 = self.activation(self.layer1(x))
            h2 = self.activation(self.layer2(h1))
            h3 = self.activation(self.layer3(h2))

        return self.layer4(h3)
