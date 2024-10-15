//
// Created by Tobias on 15.10.2024.
//

#include "NetworkViewer.h"

NetworkViewer::NetworkViewer(sf::Vector2i size, std::string title, NeuralNetwork &network) :
sf::RenderWindow(sf::VideoMode(size.x, size.y), title), _network(network) {
    // set current time
    _currentTime = std::chrono::high_resolution_clock::now();
    _previousTime = _currentTime;

    // FONT
    if (!_globalFont.loadFromFile("../../resources/fonts/Roboto.ttf")) {
        return;
    }
}

void NetworkViewer::update() {
    // time calculations
    _currentTime = std::chrono::high_resolution_clock::now();
    _deltaTime = std::chrono::duration<float>(_currentTime - _previousTime).count();
    _previousTime = _currentTime;
    _framesPerSecond = 1.0 / _deltaTime;

    sf::Event event;

    while(pollEvent(event)){
        handleEvents(event);
    }
}

void NetworkViewer::render(bool showGUI) {
    this->clear();

    renderNetwork();
    if (showGUI) renderHUD();

    this->display();
}

void NetworkViewer::renderHUD() {
    sf::Text hudText;
    hudText.setFont(_globalFont);
    hudText.setCharacterSize(16);
    hudText.setFillColor(sf::Color::White);

    std::vector<std::string> hudInfo;
    hudInfo.push_back("FPS: " + std::to_string(_framesPerSecond));
    hudInfo.push_back("Frame Time: " + std::to_string(_deltaTime * 1000.0f) + " ms");
    hudInfo.push_back("Arrayfire initialized: " + std::string(Utility::initialized() ? "true" : "false"));
    hudInfo.push_back("Device: " + std::string(Utility::deviceName()));
    hudInfo.push_back("Platform: " + std::string(Utility::platform()));
    hudInfo.push_back("Toolkit: " + std::string(Utility::toolkit()));
    hudInfo.push_back("Compute version: " + std::string(Utility::computeVersion()));
    hudInfo.push_back("Support for double operations(64 Bit): " + std::string(Utility::doubleSupport() ? "true" : "false"));
    hudInfo.push_back("Number of Layers: " + std::to_string(_network.size()));
    hudInfo.push_back("Memory info: (Occupied: " + Utility::sizeToString(_network.bytes()) + ")");

    float padding = 10.0f;
    float yPos = 10.0f;

    for (size_t i = 0; i < hudInfo.size(); i += 2) {
        // Left-aligned element
        hudText.setString(hudInfo[i]);
        sf::FloatRect textBoundsLeft = hudText.getLocalBounds();
        sf::RectangleShape textBackgroundLeft(sf::Vector2f(textBoundsLeft.width + 2 * padding, textBoundsLeft.height + 2 * padding));
        textBackgroundLeft.setFillColor(sf::Color(0, 0, 0, 180));  // Semi-transparent black
        hudText.setPosition(padding, yPos);
        textBackgroundLeft.setPosition(0, yPos - padding / 2);

        // Draw left-aligned element
        draw(textBackgroundLeft);
        draw(hudText);

        // Right-aligned element
        if (i + 1 < hudInfo.size()) {
            hudText.setString(hudInfo[i + 1]);
            sf::FloatRect textBoundsRight = hudText.getLocalBounds();
            sf::RectangleShape textBackgroundRight(sf::Vector2f(textBoundsRight.width + 2 * padding, textBoundsLeft.height + 2 * padding));
            textBackgroundRight.setFillColor(sf::Color(0, 0, 0, 180));  // Semi-transparent black

            float xPosRight = getSize().x - textBoundsRight.width - 2 * padding;
            hudText.setPosition(xPosRight + padding, yPos);
            textBackgroundRight.setPosition(xPosRight, yPos - padding / 2);

            // Draw right-aligned element
            draw(textBackgroundRight);
            draw(hudText);
        }

        // Increment the vertical position only once for both elements
        yPos += textBoundsLeft.height + 2 * padding;
    }
}

void NetworkViewer::renderNetwork() {
    sf::RectangleShape r;
    r.setSize({(float)this->getSize().x, (float)this->getSize().y});
    r.setFillColor(sf::Color::White);
    draw(r);
}

void NetworkViewer::handleEvents(sf::Event event) {
    switch (event.type) {
        case sf::Event::Closed:
            this->close();
            break;
        case sf::Event::Resized:
            // catch the resize events
            if (event.type == sf::Event::Resized)
            {
                // update the view to the new size of the window
                sf::FloatRect visibleArea(0, 0, event.size.width, event.size.height);
                this->setView(sf::View(visibleArea));
            }
        case sf::Event::KeyPressed:
        {

        }
        default:
            break;
    }
}
