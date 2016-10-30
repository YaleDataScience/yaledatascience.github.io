---
layout: "post"
title: "Guide to Autoencoders"
date: 2016-10-29 19:45:00
tags: autoencoders, neural networks
---

by [Krishnan Srinivasan](http://krishpop.xyz/)

(WIP)

### Useful Resources

- [VAE tutorial](https://arxiv.org/abs/1606.05908)
- [keras tutorial](https://blog.keras.io/building-autoencoders-in-keras.html)


## Autoencoders

### Introduction

We aren't going to spend too much time on just autoencoders because they are not as widely used today due to do the
development of better models. However, we will cover them because they are essential to understanding the later topics
of this guide.

The premise is this. You are trying to create a neural network that can efficiently encode your input data in a lower
dimension, which it is then able to decode back into the original input, with losing as little of the original input as
possible. The usefulness of doing this is mainly this: imagine your input data is very high dimensional, but in
reality, the only valid inputs you would ever receive are in a subspace of this high dimension. In fact, they exist in
a manifold of this space, which can be spanned using fewer dimensions, and these dimensions can have properties that
are useful to learn, as they capture some intrinsic/invariant aspect of the input space.

To achieve this dimensionality reduction, the autoencoder was introduced as an unsupervised learning way of attempting
to reconstruct a given input with fewer dimensions, which it would also learn to encode the original input into.

### Basic Architecture

Now at this point, the theory starts to involve an understanding of what neural networks are. The prototypical
autoencoder is a neural network which has input and output layers identical in width, and has the property of
"funneling" the input, after a sequence of hidden layers, into a hidden layer less wide than the input, and then
"fanning out" back to the original input dimension, and constructing the output. Typically, the sequence of layers to
the middle layer are repeated in reverse order to scale back up to the output layer. The sequence of funneling layers
are referred to as the "encoder," and the fanning out layers are called the "deocoder."

The loss function [^1] typically used in these architectures is mean squared error $J(x,z) = \lVert x - z\rVert^2$,
which measures how close the reconstructed input $z$ is to the original input $x$. When the data resembles a vector
binary values or a vector of probabilities (which are both values in the range of $[0,1]$), you can also use the
cross-entropy of reconstruction loss function, which calculates how many "bits" of information are preserved in the
reconstruction compared to the original. This loss function is $$J(x, z) = -\sum_k^d[x_k \log z_k +
(1-x_k)log(1-z_k)].$$

Once you've picked a loss function, you need to consider what activation functions to use on the hidden layers of the
autoencoder. In practice, if using the reconstructed cross-entropy as output, it is important to make sure

(a) your data is binary data/scaled from 0 to 1
(b) you are using sigmoid activation in the last layer

You can also optionally use sigmoid activations for each hidden layer, as that will keep the activation values between
0 and 1, and make it easier to perform linear transformations on the data that keeps it in the range of values that it
is provided in.

[^1]: http://www.deeplearning.net/tutorial/dA.html

### Application to pre-training networks

There are many ways to select the initial weights to a neural network architecture. A common initialization scheme is
random initialization, which sets the biases and weights of all the nodes in each hidden layer randomly, so they are in
a random point of the space, and objective function, and then find a nearby local minima using an algorithm like
[SGD](https://en.wikipedia.org/wiki/Stochastic_gradient_descent) or [Adam](https://arxiv.org/abs/1412.6980). In
2006-2007, autoencoders were discovered to be a useful way to pre-train networks (in 2012 this was applied to conv
nets), in effect initializing the weights of the network to values that would be closer to the optimal, and therefore
require less epochs to train. While I could try re-explaining how that works here, Quoc Le's explanation from his
series of Stanford lectures is much better, so I'll include the links to that below.[^2] [^3] In particular, look at
section 2.2 of the deep learning tutorial for the part about pre-training with autoencoders.

However, other random initialization schemes have been found more recently to work better than pre-training with
autoencoders. For more on this, see [Martens][HF] for Hessian-free optimization as one of these methods, and
[Sutskever, Martens et al][init survey] for an overview of initialization and momentum.

[^2]: http://www.trivedigaurav.com/blog/quoc-les-lectures-on-deep-learning/
[^3]: http://ai.stanford.edu/~quocle/tutorial2.pdf
[HF]: http://www.jmlr.org/proceedings/papers/v28/sutskever13.pdf
[init survey]: http://www.jmlr.org/proceedings/papers/v28/sutskever13.pdf

### Sparsity

One of the things that I am currently experimenting with is the construction of sparse autoencoders. These can be
implemented in a number of ways, one of which uses sparse, wide hidden layers before the middle layer to make the
network discover properties in the data that are useful for "clustering" and visualization. Typically, however, a
sparse autoencoder creates a sparse encoding by enforcing an l1 constraint on the middle layer. It does this by
including the l1 penalty in the cost function, so, if we are using MSE, the cost function becomes

$$J(x,z,s) = \lVert x - z \rVert^2 + \lambda\lVert s \rVert_1 $$

where $s$ is the sparse coding in the middle layer, and $\lambda$ is a regularization parameter that weights the
influence of the l1 constraint over the entire cost function. For more on these, see [sparse coding]

[sparse coding]: http://deeplearning.stanford.edu/wiki/index.php/Sparse_Coding:_Autoencoder_Interpretation

## Denoising Autoencoders

### Introduction

- autoencoders to reconstruct noisy data
- Useful for weight initialization
    - unsupervised learning criterion for **layer-by-layer initialization** [^4]:
        - each layer is trained to produce higher level representation
        - with successive layers, representation becomes more abstract
    - then, **global fine-tuning** of parameters with another training criterion
        - **robustness to partial destruction of input**

### Denoising Approach

- introduce noise into the observed input:
    - to yield almost the same representation
    - guided by the fact that *a good representation captures stable structures in the form of dependencies and regularities characteristic of the unknown distribution of the input*
- goal:
    - minimize *average reconstruction error* <span/>

$$
\begin{align}
\theta^*, \theta'^{*} &= \arg\min_{\theta, \theta'} \frac{1}{n} \sum_{i=1}^n L(x^i, z^i) \\
&= \arg\min_{\theta, \theta'} \frac{1}{n}\sum_{i=1}^n L(x^i, g_\theta (f_\theta (x^i)))
\end{align}
$$

- where $L$ is loss func like squared error
- An alternative loss is reconstruction cross entropy, for vectors of bit probabilities

$$
\begin{align*}
L_H(x,z) &= H(B_x \lVert B_z) \\
&= - \sum_{k=1}^d[x_klogz_k + (1-x_k)log(1-z_k)]
\end{align*}
$$

- if $x$ is a binary vector, the binary-crossentropy becomes negative log-likelihood for $x$, given by Bernoulli parameters $z$. Eq 1

### DAE objective function

- one way to destroy components of the input is by zeroing values of a random number of them. the corrupted input $\widetilde{x}$
- then, mapped with a hidden representation $y = f_\theta(\tilde{X}) = s(W\tilde{x} + b)$, and reconstruct $z = g_{\theta'}(y) = s(W'y + b')$
- define the joint distribution
$$q^0(X, \tilde{X}, Y) = q^0(X)q_D(\tilde{X}|X)\delta_{f_\theta(\tilde{X})}(Y)$$
- $\delta_u(v)$  puts mass $0$ when $u \neq v$, Y is a deterministic function of $\tilde{X}$.
- objective function minimized by SGD is:
$$\arg\min_{\theta, \theta'} \mathbb{E}_{q^0(X,\tilde{X})} L_H(X, g_{\theta'}(f_\theta(\tilde{X}))) \tag{3}$$

### Layer-wise initialization and fine-tuning

- representation of the $k$-th layer used to train $(k+1)$-th layer [^5]
    - used as initialization for network opt wrt supervised training criterion
    - greedy layer-wise approach is better than local minima than random initialization

### Practical Considerations

So, what does any of this mean? How can I use this? First, it's important to note what autoencoders are useful for. The main uses today for autoencoders are their generative and denoising capabilities, which is done with variational and denoising autoencoders. A third application is dimensionality reduction for data visualization, as autoencoders find interesting lower-dimensional embeddings of the data.

[^4]: http://www.iro.umontreal.ca/~vincentp/Publications/denoising_autoencoders_tr1316.pdf
[^5]: http://info.usherbrooke.ca/hlarochelle/publications/vincent10a.pdf

## Variational Autoencoders

To learn more about the statistical background to VAEs, [Eric Jang's post](http://blog.evjang.com/2016/08/variational-bayes.html) is a great resource to get started.

Variational Autoencoders are a relatively recent application of neural networks to generate 'samples' based on the
representations of the input space that they have 'learned.' Eric's article goes in depth into the methods that are
applied in these models, but the key take away is the goal of learning an approximation of an underlying distribution
in the data that allows you to generate samples that are close to the data input into your model. This is done by
optimizing the "encoding" $z \sim Q(Z|X)$ and "decoding" $x \sim P(X|Z)$ distributions to minimize the variational
lower bound $\mathcal{L} = \log p(x) - KL(Q(Z|X)||P(Z|X)) = \mathbb{E}_Q\big[ \log{p(x|z)} \big] - KL(Q(Z|X)||P(Z))$

## Adversarial Autoencoders

[https://arxiv.org/abs/1511.05644](https://arxiv.org/abs/1511.05644)
