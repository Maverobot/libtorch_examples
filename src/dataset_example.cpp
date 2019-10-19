#include <torch/torch.h>

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

std::ostream& operator<<(std::ostream& o, const SensorData& d) {
  o << "row_id: " << d.row_id << ", series_id: " << d.series_id
    << ", measurement_number: " << d.measurement_number << ", sensor_data: " << d.sensor_data
    << std::endl;
  return o;
}

class CustomDataset : torch::data::Dataset<CustomDataset> {
 public:
  CustomDataset(const std::string& csv_file_path){
      // read csv file to sensor_data_ and floor_types_;
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
  auto data = load_csv_file<SensorData>(argv[1]);
  std::cout << data << std::endl;
  return 0;
}
