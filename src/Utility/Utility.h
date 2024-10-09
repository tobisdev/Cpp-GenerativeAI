//
// Created by Tobias on 01.10.2024.
//

#ifndef MATURAPROJEKT_UTILITY_H
#define MATURAPROJEKT_UTILITY_H

#include <arrayfire.h>
#include <iostream>

namespace Utility{

    // Activation functions
    enum class Activations {
        ReLU, LeakyReLU, Sigmoid, Linear, Tanh
    };

    // Setup method
    void setup();

    // Calculate activation
    af::array calculate_activation(af::array &values, Activations activation, bool derivative = false);

    // Conversion functions for af::array
    static af::array vectorToArray(std::vector<float> const &vector);
    static std::vector<float> arrayToVector(af::array const &array);
    static af::array vector2DToArray(std::vector<std::vector<float>> const &vector);
    static std::vector<std::vector<float>> arrayToVector2D(af::array const &array);

    // Conversion functions for size_t
    std::string sizeToString(size_t size);
}

#endif //MATURAPROJEKT_UTILITY_H
