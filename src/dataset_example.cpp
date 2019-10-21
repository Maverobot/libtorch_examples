#include <torch/torch.h>

#include <algorithm>
#include <fstream>
#include <sstream>
#include <vector>

template <typename T>
std::ostream& operator<<(std::ostream& o, std::vector<T> data) {
  std::copy(data.cbegin(), data.cend(), std::ostream_iterator<T>(o, " "));
  return o;
}

// Parses a token from a csv line
template <typename T>
auto parse_token(std::istringstream& ss, char sep = ',') -> T {
  T result;
  std::string token;
  std::getline(ss, token, sep);
  std::stringstream stoken(token);
  stoken >> result;
  return result;
}

// Loads a csv file
template <typename T>
auto load_csv_file(const std::string& csv_path, bool has_header = true) -> std::vector<T> {
  std::vector<T> items;
  std::ifstream data(csv_path);
  std::string line;
  if (has_header)
    std::getline(data, line);
  while (std::getline(data, line)) {
    items.emplace_back(line);
  }
  return items;
}

// Custom sensor data
struct SensorData {
  SensorData() = default;
  SensorData(const std::string& csv_line) {
    std::istringstream iss(csv_line);
    row_id = parse_token<decltype(row_id)>(iss);
    series_id = parse_token<decltype(series_id)>(iss);
    measurement_number = parse_token<decltype(measurement_number)>(iss);
    while (!iss.eof()) {
      sensor_data.push_back(parse_token<decltype(sensor_data)::value_type>(iss));
    }
  }
  std::string row_id;
  int series_id;
  int measurement_number;
  std::vector<float> sensor_data;

  torch::Tensor toTensor() const {
    return torch::tensor(torch::ArrayRef<float>(sensor_data.data(), sensor_data.size())).clone();
  }
};

struct FloorType {
  FloorType() = default;
  FloorType(const std::string& csv_line) {
    std::istringstream iss(csv_line);
    series_id = parse_token<decltype(series_id)>(iss);
    group_id = parse_token<decltype(group_id)>(iss);
    surface = parse_token<decltype(surface)>(iss);
  };

  torch::Tensor toTensor(const std::vector<std::string>& surfaces) const {
    auto iter = std::find(surfaces.cbegin(), surfaces.end(), surface);
    if (iter == surfaces.end()) {
      throw std::logic_error("the surfaces must contain the FloorType::surface");
    }
    long id = iter - surfaces.begin();
    return torch::one_hot(torch::tensor({id}, torch::TensorOptions(torch::kLong)), surfaces.size());
  };

  int series_id;
  int group_id;
  std::string surface;
};

std::ostream& operator<<(std::ostream& o, const SensorData& d) {
  o << "row_id: " << d.row_id << ", series_id: " << d.series_id
    << ", measurement_number: " << d.measurement_number << ", sensor_data: " << d.sensor_data
    << std::endl;
  return o;
}

class CustomDataset : torch::data::Dataset<CustomDataset> {
 public:
  CustomDataset(const std::string& x_train_csv, const std::string& y_train_csv) {
    // read csv file to sensor_data_ and floor_types_;
    auto sensor_data = load_csv_file<SensorData>(x_train_csv);
    auto floor_types = load_csv_file<FloorType>(y_train_csv);
  };

  torch::data::Example<> get(size_t index) override {
    return {sensor_data_.at(index).clone(), floor_types_.at(index).clone()};
  };

  torch::optional<size_t> size() const override { return floor_types_.size(); };

 private:
  std::vector<torch::Tensor> sensor_data_;
  std::vector<torch::Tensor> floor_types_;
};

int main(int argc, char* argv[]) {
  /*
    argv[1] path to X_train.csv
    argv[2] path to y_train.csv
   */

  auto x_train_raw = load_csv_file<SensorData>(argv[1]);
  auto y_train_raw = load_csv_file<FloorType>(argv[2]);
  std::cout << "sensor data number: " << x_train_raw.size() << std::endl;
  std::cout << "floor types number: " << y_train_raw.size() << std::endl;

  std::vector<std::string> floor_types;
  for_each(y_train_raw.cbegin(), y_train_raw.cend(), [&floor_types](const FloorType& d) {
    if (std::find(floor_types.cbegin(), floor_types.cend(), d.surface) == floor_types.end()) {
      floor_types.push_back(d.surface);
    }
  });
  std::sort(floor_types.begin(), floor_types.end());
  std::cout << "floor types: " << floor_types << std::endl;

  std::cout << y_train_raw.at(0).surface << ": " << y_train_raw.at(0).toTensor(floor_types)
            << std::endl;
  std::cout << y_train_raw.at(1).surface << ": " << y_train_raw.at(1).toTensor(floor_types)
            << std::endl;
  std::cout << y_train_raw.at(4).surface << ": " << y_train_raw.at(4).toTensor(floor_types)
            << std::endl;
  return 0;
}
