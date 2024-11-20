//
// Created by Tobias on 05.11.2024.
//

#ifndef KI_DRAWINGAPP_H
#define KI_DRAWINGAPP_H

#include "SFML/Graphics.hpp"
#include "../../NeuralNetwork/NeuralNetwork.h"

enum Color : int {Red = 0, Blue = 1};
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
            case Red:
                return sf::Color::Red;
            case Blue:
                return sf::Color::Blue;
        }
    }
};

class DrawingApp : public sf::RenderWindow {
private:
    sf::Font _globalFont;
    sf::Vector2i _previousSize;

    int _batchSize = 80000; // For rendering
    NeuralNetwork &_network;

    std::vector<Point> _points;
    std::vector<af::array> _positions; // X and Y as float

    bool _showResult = false;
    Color _currentDrawingColor = Red;

    int _gridSize;

    void renderPoints();
    void renderNetworkOutput();
    void handleEvents(sf::Event event);

public:
    DrawingApp(sf::Vector2i size, std::string title, NeuralNetwork &_network);

    void update();
    void render(bool showHUD = true);
};


#endif //KI_DRAWINGAPP_H
