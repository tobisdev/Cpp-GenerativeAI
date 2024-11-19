//
// Created by Tobias on 01.10.2024.
//

#ifndef MATURAPROJEKT_UTILITY_H
#define MATURAPROJEKT_UTILITY_H

#include <arrayfire.h>
#include <iostream>
#include <algorithm>

class Utility{
private:
    static bool _initialized;
    static char _deviceName[64];
    static char _platform[10];
    static char _toolkit[64];
    static char _computeVersion[10];
    static af::Backend _backend;
    static bool _doubleSupport;
    static int _availableDevices;

public:
    // Activation functions
    enum class Activations {
        ReLU, LeakyReLU, Sigmoid, Linear, Tanh
    };

    // Setup method
    static void setup();

    // Calculate activation
    static af::array calculate_activation(af::array &values, Activations activation, bool derivative = false);

    // Conversion functions for af::array
    static af::array vectorToArray(std::vector<float> const &vector);
    static std::vector<float> arrayToVector(af::array const &array);
    static af::array vector2DToArray(std::vector<std::vector<float>> const &vector);
    static std::vector<std::vector<float>> arrayToVector2D(af::array const &array);

    // Conversion functions for size_t
    static std::string sizeToString(size_t size);

    // Find the n biggest values in a vector
    static std::vector<int> find_top_n(const std::vector<float>& vec, int n);

    // Map results
    static int mapVectorToIndex(std::vector<float> const &vector);
    static std::vector<float> mapIndexToVector(int index, int size);
    static int mapArrayToIndex(af::array const &array);
    static af::array mapIndexToArray(int index, int size);

    static af::array mapArrayToIndices(const af::array& input); // Chat abi

    // Getter and setter
    [[nodiscard]] static bool &initialized() { return _initialized; }
    [[nodiscard]] static char* deviceName() { return _deviceName; }
    [[nodiscard]] static char* platform() { return _platform; }
    [[nodiscard]] static char* toolkit() { return _toolkit; }
    [[nodiscard]] static char* computeVersion() { return _computeVersion; }
    [[nodiscard]] static af::Backend &backend() { return _backend; }
    [[nodiscard]] static bool &doubleSupport() { return _doubleSupport; }
    [[nodiscard]] static int availableDevices() { return _availableDevices; };
    static void initialized(const bool &value) { _initialized = value; }
    static void dDeviceName(const char *value) { strncpy(_deviceName, value, sizeof(_deviceName) - 1); _deviceName[sizeof(_deviceName) - 1] = '\0'; }
    static void platform(const char *value) { strncpy(_platform, value, sizeof(_platform) - 1); _platform[sizeof(_platform) - 1] = '\0'; }
    static void toolkit(const char *value) { strncpy(_toolkit, value, sizeof(_toolkit) - 1); _toolkit[sizeof(_toolkit) - 1] = '\0'; }
    static void computeVersion(const char *value) { strncpy(_computeVersion, value, sizeof(_computeVersion) - 1); _computeVersion[sizeof(_computeVersion) - 1] = '\0'; }
    static void backend(const af::Backend &value) { _backend = value; }
    static void doubleSupport(const bool &value) { _doubleSupport = value; }
    static void availableDevices(const int value) { _availableDevices = value; }
};

#endif //MATURAPROJEKT_UTILITY_H
