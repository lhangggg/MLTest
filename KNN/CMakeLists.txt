cmake_minimum_required(VERSION 3.18)
project(knn)
set(CMAKE_CXX_STANDARD 11)

#requires opencv
set(OpenCV_DIR "D:/OpenCV4.5.0/opencv/build")
find_package(OpenCV REQUIRED)
message("OpenCV version: ${OpenCV_version}")
include_directories(${OpenCV_version})
link_directories(${OpenCV_LIB_DIR})

add_executable(${PROJECT_NAME} knn.cpp)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

