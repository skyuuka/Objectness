project(BING_linux)
cmake_minimum_required(VERSION 2.8)
find_package(OpenMP REQUIRED)

# OPENCV
find_package( OpenCV REQUIRED )
if(OpenCV_FOUND)
 include_directories( ${OpenCV_INCLUDE_DIRS} )
endif( OpenCV_FOUND )
list( APPEND CMAKE_CXX_FLAGS "-std=c++0x ${CMAKE_CXX_FLAGS} -fopenmp -ftree-vectorize")

add_executable(${PROJECT_NAME} main.cpp CmShow.cpp kyheader.cpp CmFile.cpp FilterTIG.cpp DataSetVOC.cpp Objectness.cpp) 
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall")

add_subdirectory (LibLinear)
set(EXTRA_LIBS ${EXTRA_LIBS} LIBLINEAR)
target_link_libraries(${PROJECT_NAME} opencv_core opencv_imgproc opencv_highgui ${EXTRA_LIBS})
