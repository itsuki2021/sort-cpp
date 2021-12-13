cmake_minimum_required(VERSION 3.0.2)
project(sort)

set(CMAKE_CXX_STANDARD 11)
set(OpenCV_DIR /usr/local/share/OpenCV/)

find_package(OpenCV REQUIRED)

add_library(${PROJECT_NAME} SHARED sort.cpp kalman_box_tracker.cpp hungarian.cpp)

add_executable(demo_${PROJECT_NAME} main.cpp)
target_link_libraries(demo_${PROJECT_NAME} ${PROJECT_NAME} ${OpenCV_LIBRARIES})