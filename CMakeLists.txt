cmake_minimum_required(VERSION 3.0.2)
project(sort)

set(CMAKE_CXX_STANDARD 17)

# opencv
set(OpenCV_DIR /usr/local/share/OpenCV/)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})
add_definitions(${OpenCV_DEFINITIONS})

add_library(${PROJECT_NAME} SHARED sort.cpp kalman_box_tracker.cpp kuhn_munkres.cpp)

add_executable(demo_${PROJECT_NAME} main.cpp)
target_link_libraries(demo_${PROJECT_NAME} ${PROJECT_NAME} ${OpenCV_LIBRARIES})