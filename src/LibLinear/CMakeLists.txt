add_subdirectory (blas)
add_library (LIBLINEAR linear.cpp train.c tron.cpp)
target_link_libraries (LIBLINEAR BLAS)
