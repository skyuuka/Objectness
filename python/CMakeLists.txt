find_package(PythonInterp REQUIRED)
find_package(PythonLibs REQUIRED)
find_package(Boost COMPONENTS python REQUIRED)

include_directories(${Boost_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS})
include_directories(opencv-ndarray-conversion)

add_subdirectory (opencv-ndarray-conversion)

python_add_module(cBing bing.cpp)

target_link_libraries(cBing ${Boost_LIBRARIES} ${PYTHON_LIBRARIES} 
    opencv_ndarray_conversion
    bing_objectness_core bing_objectness_liblinear
    opencv_core opencv_imgproc opencv_legacy)
