# libtorch_examples
This repository contains examples of libtorch, which is C++ front end of PyTorch.
* `hello_world.cpp`: A simple example of libtorch.
* `function_approx.cpp`: A feedforward network based function approximator, which trains on `y = cos(x)`.

## Libtorch installation
To install libtorch, it is very straightforward. You just need to go to [pytorch.org](https://pytorch.org/), download the installation zip file for C++, and decompress the file to a chosen location.
For convenience, the download links are provided here.

* Download here (Pre-cxx11 ABI):
https://download.pytorch.org/libtorch/cu100/libtorch-shared-with-deps-1.2.0.zip

* Download here (cxx11 ABI):
https://download.pytorch.org/libtorch/cu100/libtorch-cxx11-abi-shared-with-deps-1.2.0.zip

## Compilation
```bash
mkdir build && cd build
cmake .. -DLIBTORCH_PATH=${PATH_TO_DECOMPRESSED_LIBTORCH_FOLDER} -DCMAKE_BUILD_TYPE=Release
make
```
