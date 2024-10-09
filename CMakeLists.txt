cmake_minimum_required(VERSION 3.26)
project(Maturaprojekt)

# Set C++ standard
set(CMAKE_CXX_STANDARD 23)

# Find ArrayFire
find_package(ArrayFire REQUIRED)

# Glob all .cpp and .h files in src directory
file(GLOB_RECURSE SOURCES "src/*.cpp" "src/*.h")

# Add the source files to your project
add_executable(Maturaprojekt ${SOURCES}
        src/Utility/Utility.cpp
        src/Utility/Utility.h
        src/NeuralNetwork/NeuralNetwork.cpp
        src/NeuralNetwork/NeuralNetwork.h)

# Include directories
include_directories(${AF_PATH}/include)

# Link directories (if necessary)
link_directories(${AF_PATH}/lib)

# Link the ArrayFire and PDCurses libraries
target_link_libraries(Maturaprojekt ${ArrayFire_LIBRARIES})

# Define output directory for the executable
set_target_properties(Maturaprojekt PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
