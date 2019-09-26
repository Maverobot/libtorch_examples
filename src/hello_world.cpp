#include <torch/torch.h>
#include <iostream>

template <typename T>
void pretty_print(const std::string& info, T&& data) {
  std::cout << info << std::endl;
  std::cout << data << std::endl << std::endl;
}

int main() {
  // Create an eye tensor
  torch::Tensor tensor = torch::eye(3);
  pretty_print("Eye tensor: ", tensor);

  // Tensor view is like reshape in numpy, which changes the dimension representation of the tensor
  // without touching its underlying memory structure.
  tensor = torch::range(1, 9, 1);
  pretty_print("Tensor range 1x9: ", tensor);
  pretty_print("Tensor view 3x3: ", tensor.view({3, 3}));
  pretty_print("Tensor view 3x3 with D0 and D1 transposed: ", tensor.view({3, 3}).transpose(0, 1));
  tensor = torch::range(1, 27, 1);
  pretty_print("Tensor range 1x27: ", tensor);
  pretty_print("Tensor view 3x3x3: ", tensor.view({3, 3, 3}));
  pretty_print("Tensor view 3x3x3 with D0 and D1 transposed: ",
               tensor.view({3, 3, 3}).transpose(0, 1));
  pretty_print("Tensor view 3x1x9: ", tensor.view({3, 1, -1}));
}
