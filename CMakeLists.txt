cmake_minimum_required(VERSION 3.15.0)
project(micrograd)

set(CMAKE_CROSSCOMPILING "True")

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wpedantic -O0")

# I can use optimization to run the neural network but breaks ops
# set(CMAKE_CXX_FLAGS "-O2")

file(GLOB_RECURSE SOURCES
      # micrograd/*.cpp
      # micrograd/**/*.cpp
      # examples/*.cpp
      # demo/*.cpp
      demo/demo.cpp
      # demo/test.cpp
)

add_library(micrograd INTERFACE)
target_include_directories(micrograd INTERFACE include)

add_executable(test_executable ${SOURCES})
target_link_libraries(test_executable micrograd)
