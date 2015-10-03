add_executable(demo demo.cpp) 
target_link_libraries (demo bing_objectness_core bing_objectness_liblinear opencv_core opencv_imgproc opencv_legacy)
add_executable(predict predict.cpp) 
target_link_libraries (predict bing_objectness_core bing_objectness_liblinear opencv_core opencv_imgproc opencv_legacy)
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall")

