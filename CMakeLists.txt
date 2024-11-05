cmake_minimum_required(VERSION 3.26)
project(KI)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)

# Find ArrayFire
find_package(ArrayFire REQUIRED)

# Use FetchContent to download SFML at configure time
include(FetchContent)

# Set up SFML FetchContent
FetchContent_Declare(
        SFML
        GIT_REPOSITORY https://github.com/SFML/SFML.git
        GIT_TAG 2.6.1  # The version of SFML you want to use
)

# Make SFML available
FetchContent_MakeAvailable(SFML)

# Glob all .cpp and .h files in src directory
file(GLOB_RECURSE SOURCES "src/*.cpp" "src/*.h")

# Add the source files to your project
add_executable(KI ${SOURCES})

# Include directories
include_directories(${AF_PATH}/include)

# Link directories (if necessary)
link_directories(${AF_PATH}/lib)

# Link the ArrayFire and SFML libraries
target_link_libraries(KI ${ArrayFire_LIBRARIES} sfml-system sfml-window sfml-graphics)

# Define output directory for the executable
set_target_properties(KI PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Path to the folder where additional DLLs are stored
set(ADDITIONAL_DLL_PATH "path/to/dlls")  # Modify this to the correct path

# Copy SFML DLLs to the output directory
add_custom_command(TARGET KI POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_RUNTIME_DLLS:KI> $<TARGET_FILE_DIR:KI>
        COMMAND_EXPAND_LISTS)

# Find MinGW libraries if applicable
if (MINGW)

    # Extract the directory where the g++ (MinGW) compiler resides
    get_filename_component(MINGW_BIN_DIR "${CMAKE_CXX_COMPILER}" DIRECTORY)

    add_custom_command(TARGET KI POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${MINGW_BIN_DIR}/libstdc++-6.dll"
            "${MINGW_BIN_DIR}/libgcc_s_seh-1.dll"
            "${MINGW_BIN_DIR}/libwinpthread-1.dll"
            $<TARGET_FILE_DIR:KI>)
endif()