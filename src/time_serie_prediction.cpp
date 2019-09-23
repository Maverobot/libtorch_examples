#include <torch/torch.h>

int main(int /*argc*/, char* /*argv*/[]) {
  // Use GPU when present, CPU otherwise.
  torch::Device device(torch::kCPU);
  if (torch::cuda::is_available()) {
    device = torch::Device(torch::kCUDA);
    std::cout << "CUDA is available! Training on GPU." << std::endl;
  }

  const size_t kSequenceLen = 3;
  const size_t kInputDim = 1;
  const size_t kHiddenDim = 1;
  const size_t kOuputDim = 1;
  auto time_serie_detector = torch::nn::LSTM(torch::nn::LSTMOptions(kInputDim, kHiddenDim)
                                                 .dropout(0.2)
                                                 .layers(kSequenceLen)
                                                 .bidirectional(false));
  time_serie_detector->to(device);
  std::cout << time_serie_detector << std::endl;

  torch::Tensor input = torch::empty({kSequenceLen, kInputDim});
  auto input_acc = input.accessor<float, 2>();
  size_t count = 0;
  for (float i = 0.1; i < 0.4; i += 0.1) {
    input_acc[count][0] = i;
    count++;
  }
  input = input.toBackend(c10::Backend::CUDA);
  std::cout << "input = " << input << std::endl;
  time_serie_detector->zero_grad();
  time_serie_detector->forward(input.toBackend(c10::Backend::CUDA).view({input.size(0), 1, -1}));

  return 0;
}
