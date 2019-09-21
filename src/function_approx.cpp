#include <torch/torch.h>
#include <algorithm>
#include <random>

std::pair<std::vector<double>, std::vector<double>> getTrainingData(double i_max = 1000000,
                                                                    double x_min = -M_PI,
                                                                    double x_max = M_PI) {
  std::vector<double> x(i_max);
  std::vector<double> y(i_max);

  size_t i = 0;
  while (i < i_max) {
    x[i] = x_min + (x_max - x_min) * i / i_max;
    y.at(i) = std::cos(x.at(i));
    i++;
  }

  return std::make_pair(x, y);
};

class OneDimMappingDataset : public torch::data::Dataset<OneDimMappingDataset> {
 private:
  std::vector<double> states_, labels_;

 public:
  explicit OneDimMappingDataset(const std::vector<double>& states,
                                const std::vector<double>& labels)
      : states_(states), labels_(labels) {
    if (states.size() != labels.size()) {
      throw std::invalid_argument("states and labels must have the same length.");
    }
  };
  torch::data::Example<> get(size_t index) override {
    torch::Tensor state = torch::eye(1) * states_.at(index);
    torch::Tensor label = torch::eye(1) * labels_.at(index);
    return {state, label};
  };

  torch::optional<size_t> size() const override { return states_.size(); }
};

using namespace torch;

int main(int argc, char* argv[]) {
  const size_t epoch_size = 100;
  const size_t batch_size = 10000;
  const int64_t kLogInterval = 10;
  const int64_t kCheckpointEvery = 10000;

  // Use GPU when present, CPU otherwise.
  torch::Device device(torch::kCPU);
  if (torch::cuda::is_available()) {
    device = torch::Device(torch::kCUDA);
    std::cout << "CUDA is available! Training on GPU." << std::endl;
  }

  // Get training data.
  std::vector<double> train_x;
  std::vector<double> train_y;
  std::tie(train_x, train_y) = getTrainingData();

  std::cout << "train_x size: " << train_x.size() << std::endl;
  std::cout << "train_y size: " << train_y.size() << std::endl;

  // Generate a dataset
  auto data_set = OneDimMappingDataset(train_x, train_y).map(torch::data::transforms::Stack<>());

  const size_t data_size = data_set.size().value();
  const int64_t batches_per_epoch = std::ceil(data_size / static_cast<double>(batch_size));
  std::cout << "data size: " << data_size << std::endl;

  // Generate a data loader.
  auto data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
      std::move(data_set), batch_size);

  // Define network
  nn::Sequential func_approximator(
      nn::Linear(nn::LinearOptions(1, 100).with_bias(true)), nn::Functional(torch::leaky_relu, 0.2),
      nn::Linear(nn::LinearOptions(100, 1).with_bias(true)), nn::Functional(torch::tanh));
  func_approximator->to(device);

  // Define Optimizer
  optim::Adam optimizer(func_approximator->parameters(), optim::AdamOptions(2e-4).beta1(0.5));

  const bool kRestoreFromCheckpoint = false;
  if (kRestoreFromCheckpoint) {
    torch::load(func_approximator, "cos-func-approx-checkpoint.pt");
  }

  size_t epoch_idx = 0;
  while (epoch_idx < epoch_size) {
    size_t batch_index = 0;
    for (torch::data::Example<>& batch : *data_loader) {
      // Train discriminator with real images.
      func_approximator->zero_grad();
      auto data = batch.data.to(device);
      // std::cout << data.to(TensorOptions().device(kCPU), false,
      // true)
      //           << std::endl;
      auto labels = batch.target.to(device);

      // std::cout << labels.to(TensorOptions().device(kCPU), false,
      // true)
      //           << std::endl;

      torch::Tensor real_output = func_approximator->forward(data);
      torch::Tensor d_loss_real = torch::mse_loss(real_output, labels);
      d_loss_real.backward();
      optimizer.step();

      if (batch_index % kLogInterval == 0) {
        std::printf("\r[%2ld/%2ld][%3ld/%3ld] loss: %.4f \n", epoch_idx, epoch_size, batch_index,
                    batches_per_epoch, d_loss_real.item<double>());
        /*
        auto test_x = torch::randn(1) * M_PI;
        auto test_y =
            func_approximator->forward(test_x.toBackend(c10::Backend::CUDA));
        std::printf("x = %.5f, target y = %.5f, predicted y = %.5f\n ",
                    test_x[0].item<double>(),
                    std::cos(test_x[0].item<double>()),
                    test_y[0].item<double>());
        */
      }

      if (batch_index % kCheckpointEvery == 0) {
        // Checkpoint the model and optimizer state.
        torch::save(func_approximator, "cos-func-approx-checkpoint.pt");
      }
      batch_index++;
    }
    epoch_idx++;
  }

  return 0;
}
