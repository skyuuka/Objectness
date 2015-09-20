cmake_minimum_required(VERSION 2.8)
project(BingObjectness)
# the version number
set (BingObjectness_VERSION_MAJOR 1)
set (BingObjectness_VERSION_MINOR 0)

find_package(OpenMP REQUIRED)
# OPENCV
find_package( OpenCV REQUIRED )
if(OpenCV_FOUND)
 include_directories( ${OpenCV_INCLUDE_DIRS} )
endif( OpenCV_FOUND )
list( APPEND CMAKE_CXX_FLAGS "-std=c++0x ${CMAKE_CXX_FLAGS} -fopenmp -ftree-vectorize")

include_directories( "include" )
add_subdirectory (src)
add_subdirectory (tools)
