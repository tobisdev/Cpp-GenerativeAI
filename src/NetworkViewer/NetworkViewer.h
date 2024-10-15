//
// Created by Tobias on 15.10.2024.
//

#ifndef KI_NETWORKVIEWER_H
#define KI_NETWORKVIEWER_H

#include "../NeuralNetwork/NeuralNetwork.h"
#include "../Utility/Utility.h"

#include <SFML/Graphics.hpp>
#include <chrono>
#include <iostream>

class NetworkViewer : public sf::RenderWindow {
private:
    float _deltaTime = 0.0;
    float _framesPerSecond = 0.0;
    std::chrono::time_point<std::chrono::high_resolution_clock> _currentTime;
    std::chrono::time_point<std::chrono::high_resolution_clock> _previousTime;

    sf::Font _globalFont;

    NeuralNetwork &_network;

    void renderHUD();
    void renderNetwork();
    void handleEvents(sf::Event event);

public:
    NetworkViewer(sf::Vector2i size, std::string title, NeuralNetwork &_network);

    void update();
    void render(bool showHUD = true);

};


#endif //KI_NETWORKVIEWER_H
