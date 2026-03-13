# libtorch_examples

Examples using [libtorch](https://pytorch.org/cppdocs/), the C++ frontend of PyTorch (v2.5.1).

## Examples

| Example                           | Description                                            |
|-----------------------------------|--------------------------------------------------------|
| `hello_world.cpp`                 | Tensor creation, reshaping, and transpose operations   |
| `simple_optimization_example.cpp` | Gradient descent — manual and with `torch::optim::SGD` |
| `function_approx.cpp`             | Feedforward network that learns `y = cos(x)`           |
| `time_serie_prediction.cpp`       | LSTM-based time series forward pass                    |
| `lstm_example.cpp`                | Bidirectional LSTM trained on a simple sequence        |
| `dataset_example.cpp`             | Custom dataset with CSV parsing and one-hot encoding   |

## Requirements

- CMake ≥ 3.14
- C++17 compiler

libtorch is downloaded automatically during the CMake configure step if not found locally.

## Compilation

Download libtorch automatically:
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release && cmake --build .
```

Use an existing local libtorch:
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/absolute/path/to/libtorch && cmake --build .
```

## Running

```bash
./build/hello_world
./build/simple_optimization_example 5
./build/function_approx
./build/dataset_example data/X_train_sample.csv data/y_train_sample.csv
```
