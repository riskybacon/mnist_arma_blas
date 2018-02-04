# MNIST using Armadillo and BLAS
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)  

This small project is an intermediate step of moving the
[mnist_arma](https://github.com/riskybacon/mnist_arma) project away from
[Armadillo](http://arma.sourceforge.net). There are no Armadillo references in
this project except for earlier commits.

Lessons learned in this project will be used in the [cakematic](https://github.com/riskybacon/cakematic) library.

Two important goals are documentation using [Sphinx](http://www.sphinx-doc.org/en/stable/)
documentation and integrating Sphinx with [Doxygen](http://www.stack.nl/~dimitri/doxygen/manual/docblocks.html).

# Dependencies:

* [Homebrew](https://brew.sh), for macOS builds
* [FreeType](https://www.freetype.org)
* [pngwriter](https://github.com/pngwriter/pngwriter)

```
brew install freetype
```

Install pngwriter from source, install into /usr/local

# Building

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

If you are using [iTerm2](https://iterm2.com), have [imagecat](https://iterm2.com/utilities/imgcat) and
[ImageMagick]([http://imagemagick.org/) installed, display the weights with:

```
convert theta1.png -resize 500 - | imagecat
convert theta2.png -resize 500 - | imagecat
```

Install ImageMagick with [homebrew](https://brew.sh):

```
brew install imagemagick
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
