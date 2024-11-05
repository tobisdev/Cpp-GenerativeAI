//
// Created by Tobias on 05.11.2024.
//

#include "DrawingApp.h"

#ifdef max
#undef max
#endif
#ifdef min
#undef min
#endif

DrawingApp::DrawingApp(sf::Vector2i size, std::string title, NeuralNetwork &network) :
sf::RenderWindow(sf::VideoMode(size.x, size.y), title), _network(network) {
    // FONT
    if (!_globalFont.loadFromFile("../../resources/fonts/Roboto.ttf")) {
        return;
    }

    _previousSize = size;

    gridSize = (size.x > size.y) ? size.y : size.x;
}

void DrawingApp::update() {
    int networks = _network.networks();

    if(!showResult){
        std::vector<float> fitness(networks, 0.0f);

        for (auto &point : points) {

            std::vector<float> pos = {
                    point.position.x / ((float)gridSize - 1.0f),
                    point.position.y / ((float)gridSize - 1.0f)
            };
            std::vector<float> col = {(float)point.color.r / 255.0f, (float)point.color.g / 255.0f, (float)point.color.b / 255.0f};

            af::array in = Utility::vectorToArray(pos);
            af::array in3d = af::tile(in, 1, 1, networks);

            af::array result3d = _network.feed_forward(in3d);

            af::array exp = Utility::vectorToArray(col);
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
        _network.breed(fitness, 1, -0.1f, 0.1f);
    }



    sf::Event event;
    while (this->pollEvent(event))
    {
        handleEvents(event);
    }
}


void DrawingApp::render(bool showHUD) {

    sf::Vector2u size = this->getSize();
    sf::Vector2f leftUpperCorner((float)size.x / 2.0f - (float)gridSize / 2.0f, (float)size.y / 2.0f - (float)gridSize / 2.0f);

    this->clear(sf::Color::Black);

    sf::RectangleShape rect;
    rect.setSize({(float)gridSize, (float)gridSize});
    rect.setFillColor({40, 40, 40});
    rect.setPosition(leftUpperCorner);

    this->draw(rect);

    if(!showResult){
        if (!points.empty())
        {
            this->draw(&points[0], points.size(), sf::Points);
        }
    }else{
//        // Generate normalized coordinate pairs
//        af::array coordinates = generate_coordinate_pairs(gridSize, gridSize) / (gridSize - 1);
//
//        af::array result = _network.feed_forward_single(coordinates, 0);
//        result = af::moddims(result, af::dim4(3, gridSize * gridSize));
//
//        // Initialize the grid of vertices
//        std::vector<sf::Vertex> grid(gridSize * gridSize);
//
//        // Transfer the result data from GPU to CPU in one call
//        std::vector<float> colors(result.elements());
//        result.host(colors.data());
//
//        for (int index = 0; index < gridSize * gridSize; ++index) {
//            int i = index % gridSize; // x-coordinate
//            int j = index / gridSize; // y-coordinate
//
//            sf::Vertex v;
//            v.position = sf::Vector2f(i, j);
//
//            // Extract RGB values
//            float r = colors[index * 3 + 0];
//            float g = colors[index * 3 + 1];
//            float b = colors[index * 3 + 2];
//
//            // Convert the normalized RGB values to sf::Color
//            sf::Color color(
//                    static_cast<sf::Uint8>(r * 255.0f),
//                    static_cast<sf::Uint8>(g * 255.0f),
//                    static_cast<sf::Uint8>(b * 255.0f)
//            );
//
//            v.color = color;
//            grid[index] = v;
//        }
//
//        // Draw the grid of points
//        sf::Transform transform;
//        transform.translate(leftUpperCorner);
//        this->draw(&grid[0], grid.size(), sf::Points, transform);

        // Set grid size and batch size
        int batchSize = 10000; // Adjust based on GPU capabilities

        // Total number of points
        int totalPoints = gridSize * gridSize;

        // Prepare vectors to hold all positions and colors
        std::vector<float> x_positions(totalPoints);
        std::vector<float> y_positions(totalPoints);

        // Generate coordinate pairs in row-major order
        for (int index = 0; index < totalPoints; ++index) {
            int x = index % gridSize;
            int y = index / gridSize;
            x_positions[index] = static_cast<float>(x) / (gridSize - 1); // Normalize to [0,1]
            y_positions[index] = static_cast<float>(y) / (gridSize - 1); // Normalize to [0,1]
        }

        // Prepare vector to hold all colors
        std::vector<float> colors(totalPoints * 3); // 3 channels (RGB)

        // Process inputs in batches
        int numBatches = (totalPoints + batchSize - 1) / batchSize; // Ceiling division

        for (int batch = 0; batch < numBatches; ++batch) {
            int startIdx = batch * batchSize;
            int endIdx = std::min(startIdx + batchSize, totalPoints);
            int currentBatchSize = endIdx - startIdx;

            // Prepare input arrays for the current batch
            std::vector<float> batch_positions(2 * currentBatchSize); // x and y positions
            for (int i = 0; i < currentBatchSize; ++i) {
                batch_positions[2 * i] = x_positions[startIdx + i];
                batch_positions[2 * i + 1] = y_positions[startIdx + i];
            }

            // Convert to ArrayFire array
            af::array in(2, 1, currentBatchSize, batch_positions.data());

            // Feed forward the batch
            af::array result = _network.feed_forward_single(in, 0); // Output shape: [3, 1, currentBatchSize]

            // Retrieve the results from the GPU
            std::vector<float> batch_colors(result.elements());
            result.host(batch_colors.data());

            // Store the colors in the main colors vector
            for (int i = 0; i < currentBatchSize; ++i) {
                colors[3 * (startIdx + i) + 0] = batch_colors[3 * i + 0]; // R
                colors[3 * (startIdx + i) + 1] = batch_colors[3 * i + 1]; // G
                colors[3 * (startIdx + i) + 2] = batch_colors[3 * i + 2]; // B
            }
        }

        // Create an sf::Image to hold the pixel data
        sf::Image image;
        image.create(gridSize, gridSize);

        // Fill the image with the colors
        for (int index = 0; index < totalPoints; ++index) {
            int x = index % gridSize;
            int y = index / gridSize;

            // Extract RGB values
            float r = colors[3 * index + 0];
            float g = colors[3 * index + 1];
            float b = colors[3 * index + 2];

            // Convert the normalized RGB values to sf::Color
            sf::Color col(
                    static_cast<sf::Uint8>(std::clamp(r, 0.0f, 1.0f) * 255.0f),
                    static_cast<sf::Uint8>(std::clamp(g, 0.0f, 1.0f) * 255.0f),
                    static_cast<sf::Uint8>(std::clamp(b, 0.0f, 1.0f) * 255.0f)
            );

            image.setPixel(x, y, col);
        }

        sf::Texture texture;
        texture.loadFromImage(image);
        sf::Sprite sprite(texture);
        this->draw(sprite);
    }

    this->display();
}

af::array DrawingApp::generate_coordinate_pairs(int x_size, int y_size) {
    // Generate x and y indices in row-major order
    af::array x_indices = af::tile(af::range(x_size), y_size); // Repeat x indices y_size times
    af::array y_indices = af::reorder(af::tile(af::range(y_size), af::dim4(1, x_size)), 1, 0); // Repeat each y index x_size times

    // Flatten y_indices to match x_indices
    y_indices = af::flat(y_indices);

    // Stack x and y indices
    af::array coords = af::join(0, x_indices, y_indices); // Shape: [2, x_size * y_size]
    coords = af::moddims(coords, af::dim4(2, 1, x_size * y_size));

    return coords;
}

sf::Color DrawingApp::value_to_color(float value, float minValue, float maxValue) {
    // Normalize the value between 0 and 1
    float normalized = (value - minValue) / (maxValue - minValue);
    normalized = std::max(0.f, std::min(1.f, normalized)); // Clamping to [0, 1]

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

void DrawingApp::handleEvents(sf::Event event) {
    sf::Vector2i mousePos = sf::Mouse::getPosition(*this);
    sf::Color drawingColor = value_to_color(currentDrawingColor, -1.0f, +1.0f);

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

        case sf::Event::MouseButtonPressed:
            if (event.mouseButton.button == sf::Mouse::Left) {
                points.push_back(sf::Vertex(sf::Vector2f(mousePos.x, mousePos.y), drawingColor));
            } else if (event.mouseButton.button == sf::Mouse::Right) {
                showResult = !showResult;
            }
            break;

        case sf::Event::MouseMoved:
            if (sf::Mouse::isButtonPressed(sf::Mouse::Left)) {
                points.push_back(sf::Vertex(sf::Vector2f(mousePos.x, mousePos.y), drawingColor));
            }
            break;

        case sf::Event::MouseWheelScrolled:

            currentDrawingColor += event.mouseWheelScroll.delta / 20.0f;

            currentDrawingColor = (currentDrawingColor > +1.0f) ? currentDrawingColor - 2.0f : currentDrawingColor;
            currentDrawingColor = (currentDrawingColor < -1.0f) ? currentDrawingColor + 2.0f : currentDrawingColor;
            break;
    }
}
