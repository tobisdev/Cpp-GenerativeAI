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
    hudText.setFont(_globalFont);  // Assuming the font is loaded correctly
    hudText.setCharacterSize(16);  // Set font size
    hudText.setFillColor(sf::Color::White);  // Default text color

    // Create colorful stats display
    std::vector<std::string> hudInfo;
    hudInfo.push_back("FPS: " + std::to_string(_framesPerSecond));
    hudInfo.push_back("Frame Time: " + std::to_string(_deltaTime * 1000.0f) + " ms");
    hudInfo.push_back("Neural Network Info:");
    hudInfo.push_back("Number of Layers: " + std::to_string(_network.size()));  // Example, assuming your NeuralNetwork class has this method
    hudInfo.push_back("Number of Neurons: 99");  // Example

    // Iterate over the HUD elements and display them with individual backgrounds
    float padding = 10.0f;
    float yPos = 10.0f;

    for (size_t i = 0; i < hudInfo.size(); ++i) {
        hudText.setString(hudInfo[i]);

        // Get the bounding box of the text
        sf::FloatRect textBounds = hudText.getLocalBounds();

        // Create a rectangle shape that fits the text
        sf::RectangleShape textBackground(sf::Vector2f(textBounds.width + 2 * padding, textBounds.height + 2 * padding));
        textBackground.setFillColor(sf::Color(0, 0, 0, 180));  // Semi-transparent black

        // Alternate between left and right alignment
        if (i % 2 == 0) {
            // Left-aligned (similar to Minecraft)
            hudText.setPosition(padding, yPos);
            textBackground.setPosition(0, yPos - padding / 2);
        } else {
            // Right-aligned (anchored to the right side of the window)
            float xPos = getSize().x - textBounds.width - 2 * padding;
            hudText.setPosition(xPos, yPos);
            textBackground.setPosition(xPos - padding, yPos - padding / 2);
        }

        // Color enhancements for FPS
        if (i == 0) {  // FPS element
            if (_framesPerSecond >= 60) {
                hudText.setFillColor(sf::Color::Green);  // Good performance
            } else if (_framesPerSecond >= 30) {
                hudText.setFillColor(sf::Color::Yellow);  // Moderate performance
            } else {
                hudText.setFillColor(sf::Color::Red);  // Poor performance
            }
        } else {
            hudText.setFillColor(sf::Color::White);  // Reset to white for other text
        }

        // Draw the background and the text
        draw(textBackground);
        draw(hudText);

        // Increment the y position for the next line
        yPos += textBounds.height + 2 * padding;
    }
}

void NetworkViewer::renderNetwork() {
    sf::RectangleShape r;
    r.setSize({800, 800});
    r.setFillColor(sf::Color::White);
    draw(r);
}

void NetworkViewer::handleEvents(sf::Event event) {

}
