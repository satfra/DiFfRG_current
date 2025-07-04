if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.31.0")
  cmake_policy(SET CMP0177 NEW)
endif()

# ##############################################################################
# Copy Python files
# ##############################################################################

install(
  DIRECTORY ${CMAKE_SOURCE_DIR}/python
  DESTINATION ./
  MESSAGE_NEVER)

# ##############################################################################
# Install the Mathematica package
# ##############################################################################

# download CPM.cmake
file(
  DOWNLOAD
  https://github.com/cpm-cmake/CPM.cmake/releases/download/v0.40.8/CPM.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/cmake/CPM.cmake
  EXPECTED_HASH
    SHA256=78ba32abdf798bc616bab7c73aac32a17bbd7b06ad9e26a6add69de8f3ae4791)
include(${CMAKE_CURRENT_BINARY_DIR}/cmake/CPM.cmake)

# Get codeparser
cpmaddpackage(
  NAME
  LibraryLinkUtilities
  GITHUB_REPOSITORY
  WolframResearch/LibraryLinkUtilities
  GIT_TAG
  v3.2.0
  DOWNLOAD_ONLY True
  )
list(APPEND CMAKE_MODULE_PATH ${LibraryLinkUtilities_SOURCE_DIR}/cmake)

find_package(WolframLanguage 12.0 COMPONENTS wolframscript)

if(${WolframLanguage_FOUND})
  message(STATUS "Wolfram Language found: ${WolframLanguage_VERSION}")
  message(STATUS "WolframScript executable: ${WolframLanguage_wolframscript_EXE}")

  # get the application directory
  execute_process(
    COMMAND
      ${WolframLanguage_wolframscript_EXE} -script ${CMAKE_CURRENT_SOURCE_DIR}/cmake/get_wolfram_app_dir.m
    OUTPUT_VARIABLE WOLFRAM_APP_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  message(STATUS "Wolfram Language application directory: ${WOLFRAM_APP_DIR}")

  # install the Mathematica package
  install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/Mathematica/DiFfRG
    DESTINATION ${WOLFRAM_APP_DIR}
    FILES_MATCHING
    PATTERN "*.m"
    PATTERN "*.wl"
    PATTERN "*.mx"
    PATTERN "*.nb"
    )
else()
  message(ERROR "Wolfram Language not found. Skipping install of Mathematica package.")
endif()