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
    // Setup the environment
    Utility::setup();

    int networks = 50000;
    std::vector<int> topology = {2, 5, 5, 2};
    std::vector<Utility::Activations> activations = {
            Utility::Activations::Tanh,
            Utility::Activations::Tanh,
            Utility::Activations::Tanh
    };

    // Initialize the neural network
    NeuralNetwork network(topology, activations, -2.8f, 2.8f, true, networks);

    std::cout << "Saving... \n";
    //network.save("testFile");
    std::cout << "Done saving!\n";

    NetworkViewer viewer({1000, 800}, "Neural-Network-Viewer", network);
    DrawingApp drawing({800, 800}, "Drawing App", network);

    viewer.setFramerateLimit(144);
    drawing.setFramerateLimit(144);

    // Stops when a window is closed
    while (drawing.isOpen() && viewer.isOpen()) {
        // This code runs every Frame
        viewer.update();
        viewer.render();
        drawing.update();
        drawing.render();
    }

    return 0;
}
