#include <torch/torch.h>
#include <cstdlib>
#include <iostream>

constexpr double kLearningRate = 0.001;
constexpr int kMaxIterations = 100000;

void native_run(double minimal) {
  // Initial x value
  auto x = torch::randn({1, 1}, torch::requires_grad(true));

  for (size_t t = 0; t < kMaxIterations; t++) {
    // Expression/value to be minimized
    auto y = (x - minimal) * (x - minimal);
    if (y.item<double>() < 1e-3) {
      break;
    }
    // Calculate gradient
    y.backward();

    // Step x value without considering gradient
    torch::NoGradGuard no_grad_guard;
    x -= kLearningRate * x.grad();

    // Reset the gradient of variable x
    x.grad().reset();
  }

  std::cout << "[native] Actual minimal x value: " << minimal
            << ", calculated optimal x value: " << x.item<double>() << std::endl;
}

void optimizer_run(double minimal) {
  // Initial x value
  std::vector<torch::Tensor> x;
  x.push_back(torch::randn({1, 1}, torch::requires_grad(true)));
  auto opt = torch::optim::SGD(x, torch::optim::SGDOptions(kLearningRate));

  for (size_t t = 0; t < kMaxIterations; t++) {
    // Expression/value to be minimized
    auto y = (x[0] - minimal) * (x[0] - minimal);
    if (y.item<double>() < 1e-3) {
      break;
    }
    // Calculate gradient
    y.backward();

    // Step x value without considering gradient
    opt.step();
    // Reset the gradient of variable x
    opt.zero_grad();
  }

  std::cout << "[optimizer] Actual minimal x value: " << minimal
            << ", calculated optimal x value: " << x[0].item<double>() << std::endl;
}

// optimize y = (x - 10)^2
int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::cout << "Usage: " << argv[0] << " minimal_value\n";
    return 1;
  }
  native_run(atof(argv[1]));
  optimizer_run(atof(argv[1]));
  return 0;
}
