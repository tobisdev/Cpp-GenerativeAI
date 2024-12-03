//
// Created by Tobias on 15.10.2024.
//

#include "NetworkViewer.h"

#ifdef max
#undef max
#endif
#ifdef min
#undef min
#endif

NetworkViewer::NetworkViewer(sf::Vector2i size, std::string title, NeuralNetwork &network) :
sf::RenderWindow(sf::VideoMode(size.x, size.y), title), _network(network) {
    // Set current time
    _currentTime = std::chrono::high_resolution_clock::now();
    _previousTime = _currentTime;

    // FONT
    if (!_globalFont.loadFromFile("../../resources/fonts/Roboto.ttf")) {
        return;
    }

    _networkView = this->getView();
    _previousSize = size;
}

void NetworkViewer::update() {
    // Time calculations
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

    // Save the current view to reset it later
    sf::View originalView = this->getView();

    // Set the view for rendering the network (the networkView is applied in handleEvents)
    this->setView(_networkView);
    renderNetwork();  // Render the network with the current view

    // Reset the view to the original (for HUD rendering)
    this->setView(originalView);
    if (showGUI) {
        renderHUD();  // Render the HUD with the fixed view
    }

    this->display();
}

void NetworkViewer::renderHUD() {
    sf::Text hudText;
    hudText.setFont(_globalFont);
    hudText.setCharacterSize(14);
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

    float padding = 6.0f;
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
    std::vector<int> topology = _network.topology();
    auto &weights = _network.weights();
    auto &biases = _network.biases();

    const float minWeight = -1.0f, maxWeight = 1.0f;
    const float minBias = -1.0f, maxBias = 1.0f;

    const float neuronRadius = 10.f;
    const float layerSpacing = 75.0f;
    const float neuronSpacing = 50.f;

    const float outline = 2.5f;

    // Cache weights and biases from ArrayFire once
    std::vector<std::vector<std::vector<float>>> cachedWeights(topology.size() - 1); // 3D array for weights
    std::vector<std::vector<float>> cachedBiases(topology.size() - 1); // 2D array for biases

    for (size_t layer = 1; layer < topology.size(); ++layer) {
        cachedWeights[layer - 1].resize(topology[layer]);
        for (int neuron = 0; neuron < topology[layer]; ++neuron) {
            cachedWeights[layer - 1][neuron].resize(topology[layer - 1]);
            for (int prevNeuron = 0; prevNeuron < topology[layer - 1]; ++prevNeuron) {
                cachedWeights[layer - 1][neuron][prevNeuron] = weights[layer - 1](neuron, prevNeuron).scalar<float>();
            }
        }
        cachedBiases[layer - 1].resize(topology[layer]);
        for (int neuron = 0; neuron < topology[layer]; ++neuron) {
            cachedBiases[layer - 1][neuron] = biases[layer - 1](neuron).scalar<float>();
        }
    }

    // Vertex array for all lines (connections between neurons)
    sf::VertexArray lines(sf::Lines);

    // Create circle shapes for neurons
    std::vector<sf::CircleShape> neurons;

    for (size_t layer = 0; layer < topology.size(); ++layer) {
        float x = (layer + 1) * layerSpacing;
        float yOffset = (this->getSize().y - topology[layer] * neuronSpacing) / 2.0f;

        // Collect connections between layers
        if (layer > 0) {
            for (int prevNeuron = 0; prevNeuron < topology[layer - 1]; ++prevNeuron) {
                float prevX = layer * layerSpacing;
                float prevY = (this->getSize().y - topology[layer - 1] * neuronSpacing) / 2.0f + prevNeuron * neuronSpacing;

                for (int currNeuron = 0; currNeuron < topology[layer]; ++currNeuron) {
                    float currX = (layer + 1) * layerSpacing;
                    float currY = (this->getSize().y - topology[layer] * neuronSpacing) / 2.0f + currNeuron * neuronSpacing;

                    sf::Color lineColor = valueToColor(cachedWeights[layer - 1][currNeuron][prevNeuron], minWeight, maxWeight);

                    // Add the line to the vertex array
                    lines.append(sf::Vertex(sf::Vector2f(prevX, prevY), lineColor));
                    lines.append(sf::Vertex(sf::Vector2f(currX, currY), lineColor));
                }
            }
        }

        // Collect neurons in the current layer as circles
        for (int neuron = 0; neuron < topology[layer]; ++neuron) {
            sf::CircleShape neuronShape(neuronRadius);
            neuronShape.setPosition(x - neuronRadius, yOffset + neuron * neuronSpacing - neuronRadius);
            neuronShape.setOutlineThickness(outline);
            neuronShape.setOutlineColor((layer > 0) ? valueToColor(cachedBiases[layer - 1][neuron], minBias, maxBias) : sf::Color::White);
            neuronShape.setFillColor(sf::Color::Black);

            // Store the neuron for later drawing
            neurons.push_back(neuronShape);
        }
    }

    // Draw all lines with one call
    this->draw(lines);

    // Draw all neurons with one call
    for (auto &neuron : neurons) {
        this->draw(neuron);
    }
}

void NetworkViewer::handleEvents(sf::Event event) {
    const float moveSpeed = 20.0f;  // Speed for moving the network with arrow keys
    const float zoomSpeed = 0.1f;   // Zoom factor for mouse wheel

    switch (event.type) {
        case sf::Event::Closed:
            this->close();
            break;

        case sf::Event::Resized:
        {
            sf::View currView = this->getView();

            sf::Vector2i diff = {(int)event.size.width - _previousSize.x, (int)event.size.height - _previousSize.y};
            _previousSize = {(int)event.size.width, (int)event.size.height};


            currView.move(diff.x / 2, diff.y / 2);
            currView.setSize(event.size.width, event.size.height);
            this->setView(currView);

            _networkView.move(diff.x / 2, diff.y / 2);
            _networkView.setSize(event.size.width, event.size.height);
            break;
        }

        case sf::Event::KeyPressed:
            // Handle arrow keys for moving the network
            if (event.key.code == sf::Keyboard::Down || event.key.code == sf::Keyboard::S) {
                _networkView.move(0, -moveSpeed);
            } else if (event.key.code == sf::Keyboard::Up || event.key.code == sf::Keyboard::W) {
                _networkView.move(0, moveSpeed);
            } else if (event.key.code == sf::Keyboard::Right || event.key.code == sf::Keyboard::D) {
                _networkView.move(-moveSpeed, 0);
            } else if (event.key.code == sf::Keyboard::Left || event.key.code == sf::Keyboard::A) {
                _networkView.move(moveSpeed, 0);
            } else if (event.key.code == sf::Keyboard::Equal) {
                _networkView.zoom(1.0f - zoomSpeed);  // Zoom in
            } else if (event.key.code == sf::Keyboard::Dash) {
                _networkView.zoom(1.0f + zoomSpeed);  // Zoom out
            }
            break;

        case sf::Event::MouseWheelScrolled:
            // Handle zooming with mouse wheel
            if (event.mouseWheelScroll.delta > 0) {
                _networkView.zoom(1.0f - zoomSpeed);  // Zoom in
            } else if (event.mouseWheelScroll.delta < 0) {
                _networkView.zoom(1.0f + zoomSpeed);  // Zoom out
            }
            break;

        case sf::Event::MouseButtonPressed:
            if (event.mouseButton.button == sf::Mouse::Left) {
                _dragging = true;
                _prevMousePos = sf::Mouse::getPosition(*this);
            }
            break;

        case sf::Event::MouseButtonReleased:
            if (event.mouseButton.button == sf::Mouse::Left) {
                _dragging = false;
            }
            break;

        case sf::Event::MouseMoved:
            if (_dragging) {
                sf::Vector2i currentMousePos = sf::Mouse::getPosition(*this);
                sf::Vector2f delta = static_cast<sf::Vector2f>(_prevMousePos - currentMousePos);
                _networkView.move(delta);
                _prevMousePos = currentMousePos;
            }
            break;

        default:
            break;
    }
}

sf::Color NetworkViewer::valueToColor(float value, float minValue, float maxValue) {
    // Normalize the value between 0 and 1
    float normalized = (value - minValue) / (maxValue - minValue);
    normalized = std::max(0.f, std::min(1.f, normalized)); // Normalizing to [0, 1]

    // Map the normalized value to a hue (0 to 360 degrees)
    float hue = normalized * 360.0f;

    // HSV to RGB conversion
    float c = 1.0f;
    float x = c * (1 - std::abs(fmod(hue / 60.0f, 2) - 1));
    float m = 0;

    float r = 0, g = 0, b = 0;
    if (hue >= 0 && hue < 60) {
        r = c, g = x, b = 0;
    } else if (hue >= 60 && hue < 120) {
        r = x, g = c, b = 0;
    } else if (hue >= 120 && hue < 180) {
        r = 0, g = c, b = x;
    } else if (hue >= 180 && hue < 240) {
        r = 0, g = x, b = c;
    } else if (hue >= 240 && hue < 300) {
        r = x, g = 0, b = c;
    } else if (hue >= 300 && hue <= 360) {
        r = c, g = 0, b = x;
    }

    // Convert to 8-bit RGB
    return sf::Color(static_cast<sf::Uint8>((r + m) * 255),
                     static_cast<sf::Uint8>((g + m) * 255),
                     static_cast<sf::Uint8>((b + m) * 255));
}