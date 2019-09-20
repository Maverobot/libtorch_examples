#include <iostream>
#include <torch/torch.h>

int main() {
  torch::Tensor tensor = torch::eye(3);
  std::cout << "Hello world from eye tensor:" << std::endl;
  std::cout << tensor << std::endl;
}
