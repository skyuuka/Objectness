add_library (bing_objectness_blas daxpy.c ddot.c dnrm2.c dscal.c)
install (TARGETS bing_objectness_blas
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    )
