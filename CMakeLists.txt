cmake_minimum_required(VERSION 3.0.2)
project(sort)

set(CMAKE_CXX_STANDARD 17)

# opencv
set(OpenCV_DIR /usr/local/share/OpenCV/)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})
add_definitions(${OpenCV_DEFINITIONS})

# include
include_directories(
    ${PROJECT_SOURCE_DIR}/include/
)

# add library from source files
file(GLOB_RECURSE SRC_FILES src/*.cpp)
add_library(${PROJECT_NAME} SHARED ${SRC_FILES})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBRARIES})

# add executable
add_executable(demo_${PROJECT_NAME} main.cpp)
target_link_libraries(demo_${PROJECT_NAME} ${PROJECT_NAME})