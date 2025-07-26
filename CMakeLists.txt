cmake_minimum_required(VERSION 3.18)
project(CudaTransformer VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/src)

file(GLOB_RECURSE CPP_SOURCES
    "src/*.cpp"
)

add_executable(cuda_transformer ${CPP_SOURCES})

set_target_properties(cuda_transformer PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Limpieza opcional
add_custom_target(distclean
    COMMAND ${CMAKE_COMMAND} -E rm -rf ${CMAKE_BINARY_DIR}/bin
)