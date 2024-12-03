//
// Created by Tobias on 05.11.2024.
//

#include "DrawingApp.h"
int enumSize = 2;

DrawingApp::DrawingApp(sf::Vector2i size, std::string title, NeuralNetwork &network) :
sf::RenderWindow(sf::VideoMode(size.x, size.y), title), _network(network) {
    // Load the font
    if (!_globalFont.loadFromFile("../../resources/fonts/Roboto.ttf")) {
        return;
    }

    _previousSize = size;
    _gridSize = (size.x > size.y) ? size.y : size.x;

    int totalPoints = _gridSize * _gridSize;

    // Prepare vectors to hold all positions
    std::vector<float> x_positions(totalPoints);
    std::vector<float> y_positions(totalPoints);

    // Generate coordinate pairs in row-major order
    for (int index = 0; index < totalPoints; ++index) {
        int x = index % _gridSize;
        int y = index / _gridSize;
        x_positions[index] = static_cast<float>(x) / (_gridSize - 1); // Normalize to [0,1]
        y_positions[index] = static_cast<float>(y) / (_gridSize - 1); // Normalize to [0,1]
    }

    int numBatches = (totalPoints + _batchSize - 1) / _batchSize; // Ceiling division
    for (int batch = 0; batch < numBatches; ++batch) {
        int startIdx = batch * _batchSize;
        int endIdx = std::min(startIdx + _batchSize, totalPoints);
        int currentBatchSize = endIdx - startIdx;

        std::vector<float> positionsVector(2 * currentBatchSize); // x and y positions
        for (int i = 0; i < currentBatchSize; ++i) {
            positionsVector[2 * i] = x_positions[startIdx + i];
            positionsVector[2 * i + 1] = y_positions[startIdx + i];
        }

        // Convert to ArrayFire array
        af::array in(2, 1, currentBatchSize, positionsVector.data());
        _positions.emplace_back(in);
    }
}

void DrawingApp::update() {
    int totalPoints = _points.size();
    int networks = _network.networks();

    // Train the network on the placed points
    if(totalPoints > 0){
        std::vector<float> fitness(networks, 0.0f);

        for (auto &point : _points) {

            std::vector<float> pos = {
                    point.position.x / ((float)_gridSize - 1.0f),
                    point.position.y / ((float)_gridSize - 1.0f)
            };
            af::array in = Utility::vectorToArray(pos);
            af::array in3d = af::tile(in, 1, 1, networks);

            af::array result3d = _network.feed_forward(in3d);

            af::array exp = Utility::mapIndexToArray(point.color, enumSize);
            af::array exp3d = af::tile(exp, 1, 1, networks);

            af::array error = af::pow(result3d - exp3d, 2);
            af::array error_per_network = af::sum(error, /*dim=*/0);
            auto res = Utility::arrayToVector(af::flat(error_per_network));
            for (int j = 0; j < res.size(); ++j) {
                fitness[j] += -1.0f * res[j];
            }
        }

        int idxBest = Utility::find_top_n(fitness, 1)[0];
        std::cout << "The best performing network is #" << idxBest << " with an error of: " << -1.0f * fitness[idxBest] << "\n";
        _network.breed(fitness, 500, -0.05f, +0.05f);
    }

    sf::Event event;
    while (this->pollEvent(event))
    {
        handleEvents(event);
    }
}


void DrawingApp::render(bool showHUD) {
    sf::Vector2u size = this->getSize();
    sf::Vector2f leftUpperCorner((float)size.x / 2.0f - (float)_gridSize / 2.0f, (float)size.y / 2.0f - (float)_gridSize / 2.0f);

    this->clear(sf::Color::Black);

    sf::RectangleShape rect;
    rect.setSize({(float)_gridSize, (float)_gridSize});
    rect.setFillColor({40, 40, 40});
    rect.setPosition(leftUpperCorner);

    this->draw(rect);

    renderNetworkOutput();
    renderPoints();

    this->display();
}

void DrawingApp::renderPoints() {
    // Draw Poins
    if (!_points.empty()) {
        sf::CircleShape c;
        c.setRadius(4);
        c.setOutlineThickness(2);
        c.setOutlineColor(sf::Color::Black);
        for (auto &point : _points) {
            c.setFillColor(point.getColor());
            c.setPosition(point.position);
            this->draw(c);
        }
    }
}

void DrawingApp::renderNetworkOutput() {
    // Total number of points
    int totalPoints = _gridSize * _gridSize;

    // Prepare vector to hold all colors
    std::vector<int> colors(totalPoints);

    // Process inputs in batches
    int numBatches = (totalPoints + _batchSize - 1) / _batchSize; // Ceiling division

    for (int batch = 0; batch < numBatches; ++batch) {
        int startIdx = batch * _batchSize;
        int endIdx = std::min(startIdx + _batchSize, totalPoints);
        int currentBatchSize = endIdx - startIdx;

        // Get the input array
        af::array &in = _positions[batch];

        // Feed forward the batch
        af::array result = _network.feed_forward_single(in, 0); // Output shape: [n, 1, _batchSize]

        // Reshape the result to [n, _batchSize] for easier processing
        result = af::moddims(result, af::dim4(result.dims(0), result.dims(2)));

        // Perform GPU-side operation to map results to color indices
        af::array colorIndices = Utility::mapArrayToIndices(result);

        // Retrieve the entire batch of colors in one call
        std::vector<int> batchColors(colorIndices.elements());
        colorIndices.host(batchColors.data());

        // Append the results to the main colors vector
        colors.insert(colors.begin() + startIdx, batchColors.begin(), batchColors.end());
    }

    // Create an sf::Image to hold the pixel data
    sf::Image image;
    image.create(_gridSize, _gridSize);

    // Fill the image with the colors
    for (int index = 0; index < totalPoints; ++index) {
        int x = index % _gridSize;
        int y = index / _gridSize;

        Point point;
        point.color = static_cast<Color>(colors[index]);
        sf::Color col = point.getColor();

        image.setPixel(x, y, col);
    }

    sf::Texture texture;
    texture.loadFromImage(image);
    sf::Sprite sprite(texture);
    this->draw(sprite);
}


void DrawingApp::handleEvents(sf::Event event) {
    sf::Vector2i mousePos = sf::Mouse::getPosition(*this);

    switch (event.type) {
        case sf::Event::Closed:
            this->close();
            break;

        case sf::Event::Resized:
        {
            sf::View currView = this->getView();

            sf::Vector2i diff = {(int) event.size.width - _previousSize.x, (int) event.size.height - _previousSize.y};
            _previousSize = {(int) event.size.width, (int) event.size.height};

            currView.move(diff.x / 2, diff.y / 2);
            currView.setSize(event.size.width, event.size.height);
            this->setView(currView);
            break;
        }

        case sf::Event::KeyPressed:
            if(event.key.code == sf::Keyboard::C || event.key.code == sf::Keyboard::R){
                _points.clear();
            }

        case sf::Event::MouseButtonPressed:
            if (event.mouseButton.button == sf::Mouse::Left) {
                _points.emplace_back(sf::Vector2f(mousePos.x, mousePos.y), _currentDrawingColor);
            } else if (event.mouseButton.button == sf::Mouse::Right) {
                _showResult = !_showResult;
            }
            break;

        case sf::Event::MouseMoved:
            if (sf::Mouse::isButtonPressed(sf::Mouse::Left)) {
                _points.emplace_back(sf::Vector2f(mousePos.x, mousePos.y), _currentDrawingColor);
            }
            break;

        case sf::Event::MouseWheelScrolled:

            int selected = static_cast<int>(_currentDrawingColor) + event.mouseWheelScroll.delta;
            selected = (selected >= enumSize) ? 0 : selected;
            selected = (selected < 0) ? enumSize - 1 : selected;
            _currentDrawingColor = static_cast<Color>(selected);

            break;
    }
}

