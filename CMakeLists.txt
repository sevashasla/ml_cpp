cmake_minimum_required(VERSION 3.0.0 FATAL_ERROR)

set(CMAKE_CXX_COMPILER "/usr/bin/g++")
add_compile_options(-fsanitize=address)
add_link_options(-fsanitize=address)
project(ML-CPP)
message(${CMAKE_CURRENT_SOURCE_DIR})


set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)

add_executable(main main.cpp)

message(${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/tqdm_like/")

target_include_directories(main INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/ml/")
target_link_libraries(main tqdm_like)
