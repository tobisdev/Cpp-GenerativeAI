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

    std::vector<std::vector<float>> input = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    std::vector<float> expected = {0, 1, 1, 0};

    std::vector<int> topology = {2, 5, 1};
    std::vector<Utility::Activations> activations = {
            Utility::Activations::LeakyReLU,
            Utility::Activations::LeakyReLU
    };

    int networks = 500;

    NeuralNetwork network(topology, activations, 0.1f, 01.9231f, true, networks);

    NetworkViewer viewer({1000, 800}, "Neural-Network-Viewer", network);

    viewer.setFramerateLimit(144);

    int cnt = 0;

    while(true){

        if(cnt == 122){
            std::vector<float> fitness(networks, 0.0f);

            for (int i = 0; i < input.size(); ++i) {
                af::array in = Utility::vectorToArray(input[i]);
                af::array in3d = af::tile(in, 1, 1, networks);

                af::array result = network.feed_forward(in3d);

                af::array expected3d = af::constant(expected[i], 1, 1, networks);

                af::array error = 1000.0f * (1 / af::pow(result - expected3d, 2));
                //af::print("MSE:",error);

                auto res = Utility::arrayToVector(af::flat(error));

                for (int j = 0; j < res.size(); ++j) {
                    fitness[j] += res[j];
                }
            }

            network.breed(fitness, 25, 0.1f, 0.8f);
            cnt = 0;
        }
        cnt++;

        viewer.update();
        //viewer.render();
    }

    return 0;
}
