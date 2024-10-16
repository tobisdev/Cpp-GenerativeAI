//
// Created by Tobias on 02.10.2024.
//

#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork(std::vector<int> &topology, std::vector<Utility::Activations> &activations, int n) {
    if(activations.size() != topology.size() - 1){
        std::cerr << "Sizes do not match!" << "\n";
        return;
    }

    _activations = activations;
    for (int i = 1; i < topology.size(); ++i) {
        int &currentNeurons = topology[i];
        int &previousNeurons = topology[i - 1];

        _weights.emplace_back(currentNeurons, previousNeurons, n);
        _biases.emplace_back(currentNeurons, 1, n);
    }
}

NeuralNetwork::NeuralNetwork(std::vector<int> &topology, std::vector<Utility::Activations> &activations, float min, float max, bool uniform, int n) {
    if(activations.size() != topology.size() - 1){
        std::cerr << "Sizes do not match!" << "\n";
        return;
    }

    _activations = activations;
    for (int i = 1; i < topology.size(); ++i) {
        int &currentNeurons = topology[i];
        int &previousNeurons = topology[i - 1];

        if(uniform){
            _weights.emplace_back(af::randu(currentNeurons, previousNeurons, n) * (max - min) + min);
            _biases.emplace_back(af::randu(currentNeurons, 1, n) * (max - min) + min);
        }else{
            _weights.emplace_back(af::randn(currentNeurons, previousNeurons, n) * (max - min) + min);
            _biases.emplace_back(af::randn(currentNeurons, 1, n) * (max - min) + min);
        }
    }
}

af::array NeuralNetwork::feed_forward(af::array &input) {
    af::array value = input;

    if (_weights.empty()) {
        std::cerr << "The network does not possess any layers!" << "\n";
        return value;
    }

    if (input.dims()[0] != _weights[0].dims()[1] && input.dims()[2] != _weights[0].dims()[2]) {
        std::cerr << "The input dimension must match the first layer's weight dimensions!" << "\n";
        return value;
    }

    for (int i = 0; i < _weights.size(); ++i) {
        value = af::matmul(_weights[i], value) + _biases[i];
        value = Utility::calculate_activation(value, _activations[i]);
    }

    return value;
}

int NeuralNetwork::size() {
    return (int)_weights.size() + 1;
}

size_t NeuralNetwork::bytes() {
    size_t size = 0;
    for (int i = 0; i < _weights.size(); ++i) {
        size += _biases[i].bytes() + _weights[i].bytes();
    }
    return size;
}

std::vector<int> NeuralNetwork::topology() {
    std::vector<int> output;

    for (int i = 0; i < _weights.size(); ++i) {
        int currentNeurons = (int)_weights[i].dims()[0];
        int previousNeurons = (int)_weights[i].dims()[1];

        if(i == 0){
            output.emplace_back(previousNeurons);
        }

        output.emplace_back(currentNeurons);
    }

    return output;
}
