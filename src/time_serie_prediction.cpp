#include <torch/torch.h>
int main(int /*argc*/, char* /*argv*/[]) {
  // Use GPU when present, CPU otherwise.
  torch::Device device(torch::kCPU);
  if (torch::cuda::is_available()) {
    device = torch::Device(torch::kCUDA);
    std::cout << "CUDA is available! Training on GPU." << std::endl;
  }

  torch::nn::Sequential time_serie_detector(
      torch::nn::RNN(torch::nn::RNNOptions(1, 10).dropout(0.2).layers(2).tanh()));
  time_serie_detector->to(device);

  std::cout << time_serie_detector << std::endl;

  //  torch::nn::RNNImpl impl;

  auto x = torch::ones(1).toBackend(c10::Backend::CUDA);
  auto a = torch::ones(10).toBackend(c10::Backend::CUDA);
  std::cout << "x = " << x << std::endl;
  std::cout << "a = " << a << std::endl;
  time_serie_detector->zero_grad();

  time_serie_detector->forward(x, a);

  return 0;
}
