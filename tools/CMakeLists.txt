add_executable(Demo main.cpp) 
target_link_libraries (Demo BingObjectness LIBLINEAR opencv_core opencv_imgproc opencv_legacy)
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall")

