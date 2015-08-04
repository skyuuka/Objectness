add_library (LIBLINEAR linear.cpp train.c tron.cpp)
add_subdirectory (blas)
target_link_libraries (LIBLINEAR BLAS)
