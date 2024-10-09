//
// Created by Tobias on 01.10.2024.
//

#include "Utility.h"

#ifdef max
#undef max
#endif

void Utility::setup() {

    std::cout << "Setup...\n\n";

    af::setBackend(AF_BACKEND_DEFAULT);

    char device_name[64];
    char platform[10];
    char toolkit[64];
    char compute_version[10];

    int bestDevice = af::getDevice();
    af::setDevice(bestDevice);

    af::deviceInfo(device_name, platform, toolkit, compute_version);

    std::cout << "-- Available Devices: " << af::getDeviceCount() << "\n";
    std::cout << "-- Selected Device:\n============================================================\n";

    // Output the device information
    std::cout << "> Device ID:         " << bestDevice << std::endl;
    std::cout << "> Device Name:       " << device_name << "\n";
    std::cout << "> Platform:          " << platform << "\n";

    af::Backend backend = af::getActiveBackend();
    switch(backend) {
        case AF_BACKEND_CPU:
            std::cout << "> Backend:           CPU\n";
            break;
        case AF_BACKEND_CUDA:
            std::cout << "> Backend:           CUDA\n";
            break;
        case AF_BACKEND_OPENCL:
            std::cout << "> Backend:           OpenCL\n";
            break;
    }
    std::cout << "> Toolkit:           " << toolkit << "\n";
    std::cout << "> Compute Version:   " << compute_version << "\n";

    bool supportsDouble = af::isDoubleAvailable(af::getDevice());
    std::cout << "> Double precision:  " << (supportsDouble ? "Available" : "Not available") << "\n";

    std::cout << "============================================================\n";

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