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
#include <cmath>

class NetworkViewer : public sf::RenderWindow {
private:
    float _deltaTime = 0.0;
    float _framesPerSecond = 0.0;
    std::chrono::time_point<std::chrono::high_resolution_clock> _currentTime;
    std::chrono::time_point<std::chrono::high_resolution_clock> _previousTime;

    sf::Vector2i _previousSize;

    sf::Font _globalFont;
    sf::View _networkView;
    bool _dragging = false;
    sf::Vector2i _prevMousePos;

    NeuralNetwork &_network;

    void renderHUD();
    void renderNetwork();
    void handleEvents(sf::Event event);

    sf::Color valueToColor(float value, float minValue, float maxValue);

public:
    NetworkViewer(sf::Vector2i size, std::string title, NeuralNetwork &_network);

    void update();
    void render(bool showHUD = true);

};


#endif //KI_NETWORKVIEWER_H
