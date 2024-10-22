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

    if (input.dims()[0] != _weights[0].dims()[1] || input.dims()[2] != _weights[0].dims()[2]) {
        std::cerr << "The input dimension must match the first layer's weight dimensions!" << "\n";
        return value;
    }

    for (int i = 0; i < _weights.size(); ++i) {

        // z = activation(weights * inputs + biases)

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

void NeuralNetwork::breed(std::vector<float> &fitness, int winners, float min, float max, bool uniform) {
    if(winners > _weights[0].dims()[2]){
        std::cerr << "The number of winners can not be higher than the number of networks!\n";
        return;
    }

    // Buffer for child networks
    std::vector<af::array> weights;
    std::vector<af::array> biases;

    // Already apply the mutation values at start
    for (int i = 0; i < _weights.size(); ++i) {
        if(uniform){
            weights.emplace_back(af::randu(_weights[i].dims()) * (max - min) + min);
            biases.emplace_back(af::randu(_biases[i].dims()) * (max - min) + min);
        }else{
            weights.emplace_back(af::randn(_weights[i].dims()) * (max - min) + min);
            biases.emplace_back(af::randn(_biases[i].dims()) * (max - min) + min);
        }
    }

    // Find the best neural networks
    auto selectedNetworks = Utility::find_top_n(fitness, winners);

    // Copy the winners into the children to preserve them
    for (int network = 0; network < selectedNetworks.size(); ++network) {
        for (int layer = 0; layer < _weights.size(); ++layer) {
            int selectedIdx = selectedNetworks[network].second;

            weights[layer](af::span, af::span, network) = _weights[layer](af::span, af::span, selectedIdx);
            biases[layer](af::span, af::span, network) = _biases[layer](af::span, af::span, selectedIdx);
        }
    }

    std::vector<std::pair<int, int>> breedingPairs;

    // Decide the breeding pairs
    int numPairs = _weights[0].dims()[2] - winners;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, selectedNetworks.size() - 1);

    for(int i = 0; i < numPairs; ++i){
        int idx1 = dis(gen);
        int idx2 = dis(gen);
        breedingPairs.emplace_back(selectedNetworks[idx1].second, selectedNetworks[idx2].second);
    }

    // Cross the values of the networks
    for (int layer = 0; layer < _weights.size(); ++layer) {
        af::dim4 wDims = _weights[layer].dims();
        af::dim4 bDims = _biases[layer].dims();

        for (int network = 0; network < numPairs; ++network) {
            af::array wMask = af::randu(wDims[0], wDims[1]) > 0.5f;
            af::array bMask = af::randu(bDims[0], bDims[1]) > 0.5f;

            int n1 = breedingPairs[network].first;
            int n2 = breedingPairs[network].second;

            weights[layer](af::span, af::span, winners + network) =
                    _weights[layer](af::span, af::span, n1) * wMask
                    + _weights[layer](af::span, af::span, n2) * (1 - wMask);
            biases[layer](af::span, af::span, winners + network) =
                    _biases[layer](af::span, af::span, n1) * bMask
                    + _biases[layer](af::span, af::span, n2) * (1 - bMask);

        }
    }

    // Copy the children into the networks
    for (int layer = 0; layer < _weights.size(); ++layer) {
        _weights[layer] = weights[layer];
        _biases[layer] = biases[layer];
    }
}