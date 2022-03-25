cmake_minimum_required(VERSION 3.0)

project(ML LANGUAGES CXX)

add_library(matrix_lib INTERFACE "./matrix.hpp")

add_executable(test test.cpp)


target_link_libraries(test INTERFACE matrix_lib)
