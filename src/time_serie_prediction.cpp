#include <torch/torch.h>

template <typename T>
void pretty_print(const std::string& info, T&& data) {
  std::cout << info << std::endl;
  std::cout << data << std::endl << std::endl;
}

int main(int /*argc*/, char* /*argv*/[]) {
  // Use GPU when present, CPU otherwise.
  torch::Device device(torch::kCPU);
  if (torch::cuda::is_available()) {
    device = torch::Device(torch::kCUDA);
    std::cout << "CUDA is available! Training on GPU." << std::endl;
  }

  const size_t kSequenceLen = 1;
  const size_t kInputDim = 1;
  const size_t kHiddenDim = 5;
  const size_t kOuputDim = 1;
  auto time_serie_detector = torch::nn::LSTM(torch::nn::LSTMOptions(kInputDim, kHiddenDim)
                                                 .dropout(0.2)
                                                 .num_layers(kSequenceLen)
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
  // (num_layers * num_directions, batch, hidden_size)
  auto h0 = torch::zeros({1, 1, (long)kHiddenDim}).to(device);
  auto c0 = torch::zeros({1, 1, (long)kHiddenDim}).to(device);
  input = input.to(device);
  std::cout << "input = " << input << std::endl;
  time_serie_detector->zero_grad();

  auto i_tmp = input.view({input.size(0), 1, -1});

  pretty_print("input: ", i_tmp);
  pretty_print("h0: ", h0);
  pretty_print("c0: ", c0);

  auto rnn_output = time_serie_detector->forward(i_tmp, std::make_tuple(h0, c0));
  pretty_print("rnn_output/output: ", std::get<0>(rnn_output));
  pretty_print("rnn_output/state (h_n): ", std::get<0>(std::get<1>(rnn_output)));
  pretty_print("rnn_output/state (c_n): ", std::get<1>(std::get<1>(rnn_output)));

  return 0;
}
