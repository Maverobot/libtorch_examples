# libtorch_examples
This repository contains examples of libtorch, which is C++ front end of PyTorch.
* `hello_world.cpp`: A simple example of libtorch.
* `function_approx.cpp`: A feedforward network based function approximator, which trains on `y = cos(x)`.

## Compilation

- Download libtorch with cmake automatically
  ```bash
  mkdir build && cd build
  cmake .. -DCMAKE_BUILD_TYPE=Release && cmake --build .
  ```

- Use an existing local libtorch
  ```bash
  mkdir build && cd build
  cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/absolute/path/to/libtorch && cmake --build .
  ```
