#include <torch/torch.h>
#include <algorithm>
#include <random>

class OneDimMappingDataset : public torch::data::Dataset<OneDimMappingDataset> {
 private:
  size_t size_;
  double x_min_;
  double x_max_;

 public:
  explicit OneDimMappingDataset(const size_t size,
                                const double x_min = -2 * M_PI,
                                const double x_max = 2 * M_PI)
      : size_(size), x_min_(x_min), x_max_(x_max){};
  torch::data::Example<> get(size_t index) override {
    torch::Tensor state = torch::rand(1) * (x_max_ - x_min_) + x_min_;
    torch::Tensor label = torch::ones(1) * std::cos(state[0].item<double>());
    return {state, label};
  };

  torch::optional<size_t> size() const override { return size_; }
};

int main(int /*argc*/, char* /*argv*/[]) {
  const bool kRestoreFromCheckpoint = true;
  const std::string kCheckPointFile = "cos-func-approx-checkpoint.pt";
  const size_t kEpochSize = 1000;
  const size_t kBatchSize = 1000;
  const int64_t kLogInterval = 10;
  const int64_t kCheckpointEvery = 10000;

  // Use GPU when present, CPU otherwise.
  torch::Device device(torch::kCPU);
  if (torch::cuda::is_available()) {
    device = torch::Device(torch::kCUDA);
    std::cout << "CUDA is available! Training on GPU." << std::endl;
  }

  // Generate a dataset
  auto data_set = OneDimMappingDataset(100000).map(torch::data::transforms::Stack<>());

  const size_t kDataSize = data_set.size().value();
  const int64_t kBatchesPerEpoch = std::ceil(kDataSize / static_cast<double>(kBatchSize));

  // Generate a data loader.
  auto data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
      std::move(data_set), kBatchSize);

  // Define network
  torch::nn::Sequential func_approximator(
      torch::nn::Linear(torch::nn::LinearOptions(1, 100).with_bias(true)),
      torch::nn::Functional(torch::leaky_relu, 0.2),
      torch::nn::Linear(torch::nn::LinearOptions(100, 10).with_bias(true)),
      torch::nn::Functional(torch::leaky_relu, 0.2),
      torch::nn::Linear(torch::nn::LinearOptions(10, 1).with_bias(true)),
      torch::nn::Functional(torch::tanh));
  func_approximator->to(device);

  // Define Optimizer
  torch::optim::Adam optimizer(func_approximator->parameters(),
                               torch::optim::AdamOptions(2e-4).beta1(0.5));

  if (kRestoreFromCheckpoint) {
    try {
      torch::load(func_approximator, kCheckPointFile);
      std::cout << kCheckPointFile << " loaded. Continue with training on the loaded weights."
                << std::endl;
    } catch (const c10::Error e) {
      std::cout << "Warning: " << e.msg_without_backtrace() << std::endl;
      std::cout << "Start training from beginning." << std::endl;
    }
  }

  size_t epoch_idx = 0;
  while (epoch_idx < kEpochSize) {
    size_t batch_index = 0;
    for (torch::data::Example<>& batch : *data_loader) {
      // Train discriminator with real images.
      func_approximator->zero_grad();
      auto data = batch.data.to(device);
      auto labels = batch.target.to(device);

      torch::Tensor real_output = func_approximator->forward(data);
      torch::Tensor d_loss_real = torch::mse_loss(real_output, labels);
      d_loss_real.backward();
      optimizer.step();

      if (batch_index % kLogInterval == 0) {
        std::printf("\r[%2ld/%2ld][%3ld/%3ld] loss: %.6f \n", epoch_idx, kEpochSize, batch_index,
                    kBatchesPerEpoch, d_loss_real.item<double>());
        /*
          auto test_x = -2 * M_PI + torch::rand(1) * 4 * M_PI;
          auto test_y = func_approximator->forward(test_x.toBackend(c10::Backend::CUDA));
          std::printf("x = %.5f, target y = %.5f, predicted y = %.5f\n ", test_x[0].item<double>(),
          std::cos(test_x[0].item<double>()), test_y[0].item<double>());
         */
      }

      if (batch_index % kCheckpointEvery == 0) {
        // Checkpoint the model and optimizer state.
        torch::save(func_approximator, kCheckPointFile);
      }
      batch_index++;
    }
    epoch_idx++;
  }

  return 0;
}
