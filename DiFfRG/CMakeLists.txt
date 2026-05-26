# ##############################################################################
# CMakeLists.txt for the DiFfRG library
# ##############################################################################

cmake_minimum_required(VERSION 3.20)

project(DiFfRG)

set(CMAKE_INSTALL_MESSAGE LAZY)

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/common.cmake)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  message(
    WARNING
      "CMAKE_INSTALL_PREFIX not set. Defaulting to $ENV{HOME}/.local/share/DiFfRG"
  )
  set(CMAKE_INSTALL_PREFIX
      $ENV{HOME}/.local/share/DiFfRG
      CACHE PATH "Install prefix" FORCE)
endif()

if(NOT DEFINED BUNDLED_DIR)
  set(BUNDLED_DIR ${CMAKE_INSTALL_PREFIX}/bundled)
  message(STATUS "BUNDLED_DIR not set, defaulting to ${BUNDLED_DIR}")
endif()

set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${BUNDLED_DIR})

# ##############################################################################
# Documentation
# ##############################################################################

option(DiFfRG_DOCUMENTATION "Build documentation" ON)

# The documentation is a single Sphinx site bundling the C++ (Doxygen via
# doxysphinx), Python (sphinx-autoapi) and Mathematica (generated from ::usage
# strings) references. It is orchestrated by documentation/build.sh, which needs
# Doxygen plus a Python 3.9–3.12 environment with the packages listed in
# documentation/requirements.txt. We degrade gracefully: if that toolchain is
# missing the project still configures and builds — only the docs are skipped.
find_package(Doxygen)
find_program(SPHINX_BUILD_EXECUTABLE sphinx-build)
find_program(DOXYSPHINX_EXECUTABLE doxysphinx)

if(DiFfRG_DOCUMENTATION AND DOXYGEN_FOUND AND SPHINX_BUILD_EXECUTABLE
   AND DOXYSPHINX_EXECUTABLE)
  message(STATUS "Documentation target configured (Doxygen + doxysphinx + Sphinx)")

  add_custom_target(
    documentation ALL
    COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/documentation/build.sh
            ${CMAKE_CURRENT_BINARY_DIR}/documentation
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/documentation
    COMMENT "Building unified documentation (C++ / Python / Mathematica)"
    VERBATIM)
  install(
    DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/documentation/html
    DESTINATION ${CMAKE_INSTALL_PREFIX}/documentation
    MESSAGE_NEVER)
  install(
    CODE "file(CREATE_LINK ${CMAKE_INSTALL_PREFIX}/documentation/html/index.html ${CMAKE_INSTALL_PREFIX}/Documentation.html SYMBOLIC)"
  )

elseif(DiFfRG_DOCUMENTATION)
  # Report exactly which tools are missing so the user knows what to supply.
  set(_doc_missing "")
  if(NOT DOXYGEN_FOUND)
    list(APPEND _doc_missing "doxygen")
  endif()
  if(NOT SPHINX_BUILD_EXECUTABLE)
    list(APPEND _doc_missing "sphinx-build")
  endif()
  if(NOT DOXYSPHINX_EXECUTABLE)
    list(APPEND _doc_missing "doxysphinx")
  endif()
  string(REPLACE ";" ", " _doc_missing_str "${_doc_missing}")
  message(WARNING
    "The 'documentation' target was NOT created: missing ${_doc_missing_str}.\n"
    "  * doxygen — install via your package manager (e.g. 'pacman -S doxygen graphviz').\n"
    "  * sphinx-build / doxysphinx — come from the Python toolchain. doxysphinx requires\n"
    "    Python 3.9-3.12 (NOT 3.13+), so create such an environment and install the deps:\n"
    "        python3.12 -m venv ${CMAKE_CURRENT_SOURCE_DIR}/documentation/.venv-docs\n"
    "        source ${CMAKE_CURRENT_SOURCE_DIR}/documentation/.venv-docs/bin/activate\n"
    "        pip install -r ${CMAKE_CURRENT_SOURCE_DIR}/documentation/requirements.txt\n"
    "    then re-run cmake with that environment ACTIVE so the tools are on PATH.\n"
    "  Or skip cmake entirely and build the docs directly:\n"
    "        bash ${CMAKE_CURRENT_SOURCE_DIR}/documentation/build.sh\n"
    "  Configure with -DDiFfRG_DOCUMENTATION=OFF to silence this warning.")
endif()

if(DiFfRG_ONLY_DOCUMENTATION)
  return()
endif()

# ##############################################################################
# Setup build system : Libraries
# ##############################################################################

add_subdirectory(${CMAKE_SOURCE_DIR}/cmake)
include(${CMAKE_SOURCE_DIR}/cmake/setup_build_system.cmake)

# ##############################################################################
# DiFfRG library and models
# ##############################################################################

include_directories(include)

add_subdirectory(include)
add_subdirectory(src)

# ##############################################################################
# Testing
# ##############################################################################

option(DiFfRG_TEST "Build tests" OFF)
if(DiFfRG_TEST)
  enable_testing()
  add_subdirectory(tests)

  # Mathematica tests — detect Wolfram executable
  set(WOLFRAM_EXE "")
  execute_process(
    COMMAND bash -c "which wolframscript"
    OUTPUT_VARIABLE WOLFRAM_EXE
    ERROR_QUIET)
  if(WOLFRAM_EXE STREQUAL "")
    execute_process(
      COMMAND bash -c "which wolfram"
      OUTPUT_VARIABLE WOLFRAM_EXE
      ERROR_QUIET)
  endif()
  if(WOLFRAM_EXE STREQUAL "")
    execute_process(
      COMMAND bash -c "which math"
      OUTPUT_VARIABLE WOLFRAM_EXE
      ERROR_QUIET)
  endif()
  if(DEFINED ENV{WOLFRAM_SCRIPT})
    set(WOLFRAM_EXE $ENV{WOLFRAM_SCRIPT})
  endif()
  string(REPLACE "\n" "" WOLFRAM_EXE "${WOLFRAM_EXE}")

  if(NOT WOLFRAM_EXE STREQUAL "")
    message(STATUS "Found Wolfram executable: ${WOLFRAM_EXE}")

    set(MATHEMATICA_TEST_RUNNER
        ${CMAKE_CURRENT_SOURCE_DIR}/Mathematica/DiFfRG/Tests/TestRunner.m)

    add_custom_target(
      mathematica-test
      COMMAND ${WOLFRAM_EXE} -script ${MATHEMATICA_TEST_RUNNER} | tee
              ${CMAKE_CURRENT_BINARY_DIR}/mathematica-test.log
      DEPENDS ${MATHEMATICA_TEST_RUNNER}
      COMMENT "Running Mathematica tests"
      VERBATIM)

    # Run a single test file or directory:
    #   make mathematica-test-single FILE=ExportTests.m
    # Note: $(FILE) is expanded by Make at build time, not by CMake at
    # configure time. VERBATIM is intentionally omitted so Make resolves it.
    add_custom_target(
      mathematica-test-single
      COMMAND ${WOLFRAM_EXE} -script ${MATHEMATICA_TEST_RUNNER} $(FILE) | tee
              ${CMAKE_CURRENT_BINARY_DIR}/mathematica-test.log
      DEPENDS ${MATHEMATICA_TEST_RUNNER}
      COMMENT "Running Mathematica test")

    add_test(
      NAME mathematica_tests
      COMMAND ${WOLFRAM_EXE} -script ${MATHEMATICA_TEST_RUNNER}
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/Mathematica/DiFfRG/Tests)
  else()
    message(STATUS "wolframscript/wolfram/math not found — Mathematica tests disabled")
  endif()
endif()

# ##############################################################################
# Packaging and installation
# ##############################################################################

include(cmake/install_bundled.cmake)
include(CMakePackageConfigHelpers)

write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/DiFfRGConfigVersion.cmake
  VERSION 1.0.0
  COMPATIBILITY SameMajorVersion)

configure_package_config_file(
  "Config.cmake.in" "DiFfRGConfig.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/DiFfRG
  PATH_VARS CMAKE_INSTALL_LIBDIR)

install(
  EXPORT DiFfRGTargets
  FILE DiFfRGTargets.cmake
  NAMESPACE DiFfRG::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/DiFfRG)

# Install Config and ConfigVersion files
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/DiFfRGConfig.cmake"
              "${CMAKE_CURRENT_BINARY_DIR}/DiFfRGConfigVersion.cmake"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/DiFfRG")
