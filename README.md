# blast-matrix

This will be renamed to hs-nnet eventually

A haskell library for building neural networks that utilizes OpenCL by default but has abstracted
the main operations to enable switching the backend as necessary.

Tries to achieve some type safety for layer dimension matching

Supports convolution, fullyconnected / dense, dropout, transposed convolution and depth based bias layers

Supports Relu, LeakyRelu, Sigmoid, Tanh activation layers

Supports Adam / Momentum / SGD / RMSProp optimizers with Demon momentum adjustment and/or weight clipping

Written to work cross platform, developed on Windows due to the haskell ecosystem failing often on windows.
Tested, but not extensively, this library is still a work in progress.

Created after working with grenade (nn library utilizing hmatrix)
