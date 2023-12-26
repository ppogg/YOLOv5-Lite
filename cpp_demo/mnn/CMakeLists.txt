# create by pogg on 2023-12-15
cmake_minimum_required(VERSION 3.5.1)
project(v5Lite-MNN)


set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "-std=c++14 -O3")

# add OpenMP DEP
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS} -fopenmp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -fopenmp")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS} -fopenmp")

# add OpenCV DEP
set(OpenCV_DIR "${PROJECT_SOURCE_DIR}/opencv-3.4.3/share/OpenCV")
find_package(OpenCV REQUIRED)
MESSAGE(STATUS "Project: ${PROJECT_NAME}")
MESSAGE(STATUS "OpenCV library status:")
MESSAGE(STATUS "    version: ${OpenCV_VERSION}")
MESSAGE(STATUS "    libraries: ${OpenCV_LIBS}")
MESSAGE(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

include_directories(${PROJECT_SOURCE_DIR}/include/)
LINK_DIRECTORIES(${PROJECT_SOURCE_DIR}/lib)

add_executable(main src/main.cpp src/utils.cpp)

target_link_libraries (main ${OpenCV_LIBS} libMNN.so libMNN_Express.so libMNNOpenCV.so -lpthread)
