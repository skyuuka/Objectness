include_directories( CMAKE_CURRENT_SOURCE_DIR )

file(GLOB CPP_FILES "*.cpp")
add_library (opencv_ndarray_conversion ${CPP_FILES}) 
target_link_libraries (opencv_ndarray_conversion
    opencv_core opencv_imgproc opencv_legacy)

install (TARGETS opencv_ndarray_conversion 
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    )
