cmake_minimum_required(VERSION 3.0.0 FATAL_ERROR)

project(ML-CPP)
message(${CMAKE_CURRENT_SOURCE_DIR})

add_executable(main main.cpp)

message(${CMAKE_CURRENT_SOURCE_DIR})

target_include_directories(main INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/nn/")
target_include_directories(main INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/layer/")
