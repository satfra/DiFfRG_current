configure_file(setup_build_system.cmake
               ${CMAKE_CURRENT_BINARY_DIR}/setup_build_system.cmake COPYONLY)
configure_file(verify_install.cmake
               ${CMAKE_CURRENT_BINARY_DIR}/verify_install.cmake COPYONLY)
install(FILES setup_build_system.cmake verify_install.cmake
        DESTINATION "${CMAKE_INSTALL_PREFIX}/cmake")
