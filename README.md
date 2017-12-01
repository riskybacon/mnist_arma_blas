# MNIST using Armadillo and BLAS
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)  

This small project is an intermediate step of moving the
[mnist_arma](https://github.com/riskybacon/mnist_arma) project away from
[Armadillo](http://arma.sourceforge.net).

Lessons learned in this project will be used in the [cakematic](https://github.com/riskybacon/cakematic) library.

Other goals were adding [Sphinx](http://www.sphinx-doc.org/en/stable/)
style documentation stubs and integrating Sphinx with [Doxygen](http://www.stack.nl/~dimitri/doxygen/manual/docblocks.html).

# Building

Dependencies: [armadillo](http://arma.sourceforge.net)

```
git clone git@github.com:riskybacon/mnist_arma_blas.git
cd mnist_arma_blas
mkdir build
cd build
../get_mnist.sh
cmake ..
make
./mnist_arma_blas
```

Tested under macOS, XCode 9

If you are using iTerm2 and have [imagecat](https://iterm2.com/utilities/imgcat)
installed, display the weights with:

```
convert theta1.png -resize 500 - | imagecat
convert theta2.png -resize 500 - | imagecat
```

# Documentation

## HTML

macOS:

```
brew install doxygen
cd docs
make refhtml
open _build/html/ref/html/index.html
```

## PDF

macOS:

```
brew cask install mactex
export PATH=${PATH}:/usr/local/texlive/2017/bin/x86_64-darwin
cd docs
make refpdf
open _build/ref/latex/refman.pdf
```
