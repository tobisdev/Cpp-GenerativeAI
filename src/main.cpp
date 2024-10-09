#include <iostream>
#include <arrayfire.h>

#include "Utility/Utility.h"
#include "NeuralNetwork/NeuralNetwork.h"

using namespace Utility;

int main() {
    Utility::setup();

    std::vector<int> topology;
    int numLayers;

    std::cout << "Enter the number of layers: ";
    std::cin >> numLayers;

    for (int i = 0; i < numLayers; ++i) {
        int neurons;
        std::cout << "Enter the number of neurons in layer " << (i + 1) << ": ";
        std::cin >> neurons;
        topology.push_back(neurons);
    }

    std::vector<Activations> activations(topology.size() - 1, Activations::LeakyReLU);

    NeuralNetwork network(topology, activations, 0.2f, 0.8f);

    std::cout << "Size: " << sizeToString(network.bytes()) << " \n";

    while(true);

    return 0;
}
