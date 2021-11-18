cmake_minimum_required(VERSION 3.17)
project(FreightFrenzyCV)
set(CMAKE_CXX_STANDARD 20)

FIND_PACKAGE(OpenCV REQUIRED)
INCLUDE_DIRECTORIES("/usr/local/Cellar/opencv/4.5.3_3/include/opencv4")
add_executable(vision vision.cpp)
TARGET_LINK_LIBRARIES(vision ${OpenCV_LIBS})