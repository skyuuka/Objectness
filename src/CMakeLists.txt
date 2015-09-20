add_subdirectory (LibLinear)

include_directories( CMAKE_CURRENT_SOURCE_DIR )

file(GLOB CPP_FILES "*.cpp")
add_library (BingObjectness ${CPP_FILES}) 
target_link_libraries (LIBLINEAR)
