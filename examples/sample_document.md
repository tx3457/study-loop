# Backpropagation overview

Backpropagation computes gradients of a neural network's loss with respect to
its parameters. A forward pass produces predictions and a loss value. The
backward pass applies the chain rule from the output layer toward earlier
layers. An optimizer then uses those gradients to update the parameters.

A learner should distinguish four stages: forward computation, loss
calculation, gradient propagation, and parameter update.
