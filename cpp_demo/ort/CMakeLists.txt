cmake_minimum_required(VERSION 2.25)
project (v5Lite)
set(CMAKE_CXX_STANDARD 11)	

set(OpenCV_DIR "/home/chenxr/Downloads/opencv-4.1.0/build")
find_package(OpenCV REQUIRED)
MESSAGE(STATUS "Project: ${PROJECT_NAME}")
MESSAGE(STATUS "OpenCV library status:")
MESSAGE(STATUS "version: ${OpenCV_VERSION}")
MESSAGE(STATUS "libraries: ${OpenCV_LIBS}")
MESSAGE(STATUS "include path: ${OpenCV_INCLUDE_DIRS}")

INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/include)
LINK_DIRECTORIES(${PROJECT_SOURCE_DIR}/lib)

# set(DEPS ${DEPS} "/usr/local/cuda/lib64/libcudart.so")

add_executable(demo main.cpp)
TARGET_LINK_LIBRARIES(demo
                      libonnxruntime.so
		       ${OpenCV_LIBS} -lpthread)



