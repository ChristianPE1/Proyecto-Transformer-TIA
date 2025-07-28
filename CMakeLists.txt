cmake_minimum_required(VERSION 3.18)
project(CudaTransformer VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/src)

file(GLOB_RECURSE CPP_SOURCES
    "src/*.cpp"
)

file(GLOB_RECURSE HEADERS
    "src/*.hpp"
)

add_executable(main ${CPP_SOURCES})


target_compile_definitions(main PRIVATE 
    DATA_DIR="${CMAKE_SOURCE_DIR}/data/"
    IMAGES_DIR="${CMAKE_SOURCE_DIR}/data/processed_images/"
    WEIGHTS_DIR="${CMAKE_SOURCE_DIR}/weights/"
)


set_target_properties(main PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Limpieza opcional
add_custom_target(distclean
    COMMAND ${CMAKE_COMMAND} -E rm -rf ${CMAKE_BINARY_DIR}/bin
)