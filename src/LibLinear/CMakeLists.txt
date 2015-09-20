add_subdirectory (blas)
add_library (bing_objectness_liblinear linear.cpp train.c tron.cpp)
target_link_libraries (bing_objectness_liblinear bing_objectness_blas)

install (TARGETS bing_objectness_liblinear
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    )
