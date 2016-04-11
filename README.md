## Introduction
Re-implement the Hinton et al. MNIST autoencoder (Science, 2006), but by using contemporary methods of training deep neural networks avoid any form of pre-training.

https://www.cs.toronto.edu/~hinton/science.pdf

## Differences from Hinton et al
* Use mean squared error (MSE) loss, instead of strange cross-entropy.
* Use batch normalization.
* Use RELU activations instead of sigmoids.
* Use stochastic gradient descent + momentum instead of conjugate gradient.

## Results
MNIST test MSE (per-batch):
- Hinton et al.: 3.0
- 200 iterations: 2.86
- 500 iterations: 2.65
