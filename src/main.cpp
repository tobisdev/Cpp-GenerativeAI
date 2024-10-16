#include <iostream>
#include <arrayfire.h>

#ifdef _WIN32
#define NOMINMAX
#endif

#include "Utility/Utility.h"
#include "NeuralNetwork/NeuralNetwork.h"
#include "NetworkViewer/NetworkViewer.h"

int main() {
    Utility::setup();

    std::vector<int> topology = {3, 10, 10, 3};
    std::vector<Utility::Activations> activations = {
            Utility::Activations::LeakyReLU,
            Utility::Activations::LeakyReLU,
            Utility::Activations::LeakyReLU
    };

    NeuralNetwork network(topology, activations, 0.2f, 0.8f, true);

    NetworkViewer viewer({800, 800}, "Neural-Network-Viewer", network);

    viewer.setFramerateLimit(144);

    while(true){
        //network = {topology, activations, 0.2f, 0.8f};
        viewer.update();
        viewer.render();
    }

    return 0;
}
