project(YOLOv5s)

cmake_minimum_required(VERSION 3.10.2)

# set(ncnn_DIR "/xxx/path/to/ncnn/xxx/lib/cmake/ncnn")

find_package(ncnn REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(ncnnv5lite v5lite-s.cpp)

target_link_libraries(ncnnv5lite ncnn ${OpenCV_LIBS})
