//
// Created by Tobias on 05.11.2024.
//

#ifndef KI_DRAWINGAPP_H
#define KI_DRAWINGAPP_H

#include "SFML/Graphics.hpp"
#include "../../NeuralNetwork/NeuralNetwork.h"

enum Color : int {Black = 0, White = 1};
extern int enumSize;

struct Point{
    Color color;
    sf::Vector2f position;

    Point() = default;
    Point(sf::Vector2f pos, Color col){
        color = col;
        position = pos;
    }

    sf::Color getColor(){
        switch (color) {
            case Black:
                return sf::Color::Black;
            case White:
                return sf::Color::White;
        }
    }
};

class DrawingApp : public sf::RenderWindow {
private:
    sf::Font _globalFont;
    sf::Vector2i _previousSize;

    NeuralNetwork &_network;

    std::vector<Point> points;

    af::array generate_coordinate_pairs(int x_size, int y_size);
    sf::Color value_to_color(float value, float minValue, float maxValue);

    void handleEvents(sf::Event event);

    bool showResult = false;
    Color currentDrawingColor = Black;

    int gridSize;

public:
    DrawingApp(sf::Vector2i size, std::string title, NeuralNetwork &_network);

    void update();
    void render(bool showHUD = true);
};


#endif //KI_DRAWINGAPP_H
