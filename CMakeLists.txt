cmake_minimum_required(VERSION 3.10)

project(WorkerPoolTest)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include_directories(${CMAKE_SOURCE_DIR})

add_executable(workerpool_test main.cpp)