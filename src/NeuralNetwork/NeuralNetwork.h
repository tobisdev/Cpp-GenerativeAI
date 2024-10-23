//
// Created by Tobias on 02.10.2024.
//

#ifndef MATURAPROJEKT_NEURALNETWORK_H
#define MATURAPROJEKT_NEURALNETWORK_H

#include <arrayfire.h>
#include <vector>
#include <random>
#include <chrono>

#include "../Utility/Utility.h"

class NeuralNetwork {
private:
    // Neural network values
    std::vector<af::array> _weights;
    std::vector<af::array> _biases;
    std::vector<Utility::Activations> _activations;

public:
    // Constructors
    NeuralNetwork() = default;
    NeuralNetwork(std::vector<int> &topology, std::vector<Utility::Activations> &activations, int n = 1);
    NeuralNetwork(std::vector<int> &topology, std::vector<Utility::Activations> &activations, float min,
                  float max, bool uniform = true, int n = 1);
    NeuralNetwork(std::string path);

    // Getter and setter
    [[nodiscard]] std::vector<af::array> &weights() { return _weights; }
    [[nodiscard]] std::vector<af::array> &biases() { return _biases; }
    [[nodiscard]] std::vector<Utility::Activations> &activationValues() { return _activations; }
    [[nodiscard]] af::array &weights(int i) { return _weights[i]; }
    [[nodiscard]] af::array &biases(int i) { return _biases[i]; }
    [[nodiscard]] Utility::Activations &activations(int i) { return _activations[i]; }

    // Functions
    bool load(std::string path);
    bool save(std::string path);
    int size();
    size_t bytes();
    std::vector<int> topology();

    af::array feed_forward(af::array &input);
    af::array feed_forward(std::vector<float> &input);

    void breed(af::array &fitness, int winners, float min, float max, bool uniform = true);
    void breed(std::vector<float> &fitness, int winners, float min, float max, bool uniform = true);
};


#endif //MATURAPROJEKT_NEURALNETWORK_H
