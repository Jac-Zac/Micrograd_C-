cmake_minimum_required(VERSION 3.15.0)
project(micrograd)

set(CMAKE_CROSSCOMPILING "True")

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_CXX_FLAGS "-O3")

file(GLOB_RECURSE SOURCES
      micrograd/*.cpp
      micrograd/**/*.cpp
      examples/training.cpp
)

add_library(micrograd INTERFACE)
target_include_directories(micrograd INTERFACE include)

add_executable(test_executable ${SOURCES})
target_link_libraries(test_executable micrograd)
