cmake_minimum_required(VERSION 3.20)
project(YOLOv5_Lite)

set(CMAKE_CXX_STANDARD 11)
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_DIR}/include
        ${PROJECT_SOURCE_DIR}/tengine-lite-v1.5-cuda/include)
link_directories(${OpenCV_DIR}/lib/
        ${PROJECT_SOURCE_DIR}/tengine-lite-v1.5-cuda/lib)

add_executable(YOLOv5_Lite main.cpp)
target_link_libraries(YOLOv5_Lite ${OpenCV_LIBS} libtengine-lite.so)
