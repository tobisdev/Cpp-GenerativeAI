//
// Created by Tobias on 05.11.2024.
//

#ifndef KI_DRAWINGAPP_H
#define KI_DRAWINGAPP_H

#include "SFML/Graphics.hpp"
#include "../../NeuralNetwork/NeuralNetwork.h"

class DrawingApp : public sf::RenderWindow {
private:
    sf::Font _globalFont;
    sf::Vector2i _previousSize;

    NeuralNetwork &_network;

    std::vector<sf::Vertex> points;

    std::vector<sf::Vertex> grid;

    af::array generate_coordinate_pairs(int x_size, int y_size);
    sf::Color value_to_color(float value, float minValue, float maxValue);

    void handleEvents(sf::Event event);

    bool showResult = false;
    float currentDrawingColor = 0.0f;

    int gridSize;

public:
    DrawingApp(sf::Vector2i size, std::string title, NeuralNetwork &_network);

    void update();
    void render(bool showHUD = true);
};


#endif //KI_DRAWINGAPP_H
