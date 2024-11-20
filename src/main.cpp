#include <iostream>
#include <arrayfire.h>

#ifdef _WIN32
#define NOMINMAX
#endif

#include "Utility/Utility.h"
#include "NeuralNetwork/NeuralNetwork.h"
#include "Interfaces/NetworkViewer/NetworkViewer.h"
#include "Interfaces/DrawingApp/DrawingApp.h"

int main() {
    Utility::setup();

    std::vector<int> topology = {2, 8, 2};
    std::vector<Utility::Activations> activations = {
            Utility::Activations::Tanh,
            Utility::Activations::Tanh
    };

    int networks = 50000;

    NeuralNetwork network(topology, activations, -2.8f, 2.8f, true, networks);

    NetworkViewer viewer({1000, 800}, "Neural-Network-Viewer", network);
    DrawingApp drawing({800, 800}, "Drawing App", network);

    viewer.setFramerateLimit(144);
    drawing.setFramerateLimit(144);

    while (drawing.isOpen() && viewer.isOpen()) {
        viewer.update();
        viewer.render();
        drawing.update();
        drawing.render();
    }

    return 0;
}
