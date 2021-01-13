# blast-matrix

This will be renamed to hs-nnet eventually

A haskell library for building neural networks that utilizes OpenCL by default but has abstracted
the main operations to enable switching the backend as necessary.

Tries to achieve some type safety for layer dimension matching

Supports convolution and fullyconnected / dense layers

Supports Relu, LeakyRelu, Sigmoid, Tanh layers

Supports Adam / Momentum / SGD / RMSProp optimizers with Demon momentum adjustment if applied

Tested on Windows but not extensively, this library is still a work in progress.

Created after working with grenade (nn library utilizing hmatrix)
