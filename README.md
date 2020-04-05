# libtorch_examples
This repository contains examples of libtorch, which is C++ front end of PyTorch.
* `hello_world.cpp`: A simple example of libtorch.
* `function_approx.cpp`: A feedforward network based function approximator, which trains on `y = cos(x)`.

## Compilation
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make
```
