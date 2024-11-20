//
// Created by Tobias on 01.10.2024.
//

#include "Utility.h"
#ifdef max
#undef max
#endif
#ifdef min
#undef min
#endif

bool Utility::_initialized = false;
char Utility::_deviceName[64] = "";
char Utility::_platform[10] = "";
char Utility::_toolkit[64] = "";
char Utility::_computeVersion[10] = "";
af::Backend Utility::_backend = af::Backend::AF_BACKEND_DEFAULT;
bool Utility::_doubleSupport = false;
int Utility::_availableDevices = 0;

void Utility::setup() {

    std::cout << "Setup...\n\n";

    af::setBackend(AF_BACKEND_DEFAULT);       // GPU Backend

    int bestDevice = af::getDevice();               // Find the best graphics device
    af::setDevice(bestDevice);                      // Select the best graphics device

    af::deviceInfo(_deviceName, _platform, _toolkit, _computeVersion);

    _backend = af::getActiveBackend();

    _doubleSupport = af::isDoubleAvailable(af::getDevice());

    _initialized = true;
}

af::array Utility::calculate_activation(af::array &values, Activations activation, bool derivative) {
    if(derivative){
        switch(activation){
            case Activations::Sigmoid:
            {
                af::array sig = 1.0f / (1.0f + af::exp(-values));
                return sig * (1.0f - sig);
            }
            case Activations::LeakyReLU:
                return af::select(values > 0, af::constant(1.0f, values.dims()), af::constant(0.1f, values.dims()));
            case Activations::ReLU:
                return af::select(values > 0, af::constant(1.0f, values.dims()), af::constant(0.0f, values.dims()));
            case Activations::Tanh:
            {
                af::array tanh_x = tanh(values);
                return 1.0f - af::pow(tanh_x, 2);
            }
            default:
                return af::constant(1.0f, values.dims());
        }
    }else{
        switch(activation) {
            case Activations::Sigmoid:
                return 1.0f / (1.0f + af::exp(-values));
            case Activations::LeakyReLU:
                return af::select(values > 0, values, 0.1f * values);
            case Activations::ReLU:
                return af::max(values, 0.0f);
            case Activations::Tanh:
                return af::tanh(values);
            default:
                return values;
        }
    }
}

std::vector<float> Utility::arrayToVector(const af::array &array) {
    std::size_t numElements = array.elements();

    std::vector<float> vector(numElements);

    array.host(vector.data());

    return vector;
}


af::array Utility::vectorToArray(const std::vector<float> &vector) {
    return af::array(vector.size(), vector.data());
}

af::array Utility::vector2DToArray(const std::vector<std::vector<float>> &vector) {
    std::size_t rows = vector.size();
    std::size_t cols = rows > 0 ? vector[0].size() : 0;

    std::vector<float> flatVector;
    flatVector.reserve(rows * cols);

    for (const auto& row : vector) {
        flatVector.insert(flatVector.end(), row.begin(), row.end());
    }

    return af::array(rows, cols, flatVector.data());
}

std::vector<std::vector<float>> Utility::arrayToVector2D(const af::array &array) {
    af::dim4 dims = array.dims();
    std::size_t rows = dims[0];
    std::size_t cols = dims[1];

    std::vector<float> flatVector(rows * cols);

    array.host(flatVector.data());

    std::vector<std::vector<float>> vector2D(rows, std::vector<float>(cols));
    for (std::size_t i = 0; i < rows; ++i) {
        std::copy(flatVector.begin() + i * cols, flatVector.begin() + (i + 1) * cols, vector2D[i].begin());
    }

    return vector2D;
}

std::string Utility::sizeToString(size_t size) {
    std::string out;
    byte unit = 0;
    byte komma = 0;

    while(size > 1000){
        int temp = size / 1000;
        komma = size / 10 - temp;
        size = temp;
        unit++;
    }

    out = std::to_string(size) + "." + std::to_string(komma);

    switch (unit) {
        case 0: out += " Byte"; break;
        case 1: out += " kB"; break;
        case 2: out += " MB"; break;
        case 3: out += " GB"; break;
        case 4: out += " TB"; break;
        case 5: out += " PB"; break;
        default: out += " undefined unit"; break;
    }

    return out;
}

std::vector<int> Utility::find_top_n(const std::vector<float>& vec, int n) {
    if (vec.empty() || n <= 0) {
        std::cerr << "The vector cannot be empty.\n";
        return {};
    }

    // Create a vector of pairs (value, index)
    std::vector<std::pair<float, int>> value_index_pairs;
    value_index_pairs.reserve(vec.size());

    for (size_t i = 0; i < vec.size(); ++i) {
        value_index_pairs.emplace_back(vec[i], static_cast<int>(i));
    }

    // Sort the pairs based on the values in descending order
    std::sort(value_index_pairs.begin(), value_index_pairs.end(),
              [](const std::pair<float, int>& a, const std::pair<float, int>& b) -> bool {
                  return a.first > b.first;
              }
    );

    n = std::min(n, static_cast<int>(value_index_pairs.size()));

    // Extract the indices of the top n elements
    std::vector<int> result;
    result.reserve(n);

    for (int i = 0; i < n; ++i) {
        result.push_back(value_index_pairs[i].second);
    }

    return result;
}

int Utility::mapVectorToIndex(const std::vector<float> &vector) {
    return find_top_n(vector, 1)[0];
}

std::vector<float> Utility::mapIndexToVector(int index, int size) {
    if(index >= size){
        std::cerr << "The index cannot be greater then or equal to the size.\n";
        return {};
    }

    std::vector<float> vector(size, 0.0f);
    vector[index] = 1.0f;
    return vector;
}

int Utility::mapArrayToIndex(const af::array &array) {
    return mapVectorToIndex(arrayToVector(array));
}

af::array Utility::mapIndexToArray(int index, int size) {
    return vectorToArray(mapIndexToVector(index, size));
}

af::array Utility::mapArrayToIndices(const af::array& input) {
    // Check if input has the correct number of dimensions
    if (input.numdims() != 2) {
        std::cerr << "Error: Input array must have exactly 2 dimensions.\n";
        return af::array();
    }

    // Determine the index of the highest value for each column (batch)
    af::array maxValues, indices;
    af::max(maxValues, indices, input, 0);

    // Return the indices of the maximum values
    return indices;
}