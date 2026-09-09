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
# Copy the Julia package
# ##############################################################################

install(
  DIRECTORY ${CMAKE_SOURCE_DIR}/julia
  DESTINATION ./
  MESSAGE_NEVER
  PATTERN "Manifest.toml" EXCLUDE)

# ##############################################################################
# Copy the frontend type table
# ##############################################################################
#
# schemes.toml describes which C++ types a scheme is spelled with, which headers
# it needs and in which order. Code generators read it instead of hardcoding the
# vocabulary, so it has to travel with the install.

install(
  DIRECTORY ${CMAKE_SOURCE_DIR}/frontend
  DESTINATION ./
  MESSAGE_NEVER)

# ##############################################################################
# Install the Mathematica package (optional)
# ##############################################################################
#
# The Mathematica/Wolfram Language package is optional and does not belong in
# the compiled-library install unless the user explicitly chooses a destination.
# In particular, configuration must never launch Wolfram just to discover its
# user application directory.
set(
  DiFfRG_MATHEMATICA_INSTALL_DIR
  ""
  CACHE PATH
        "Optional destination for the DiFfRG Mathematica package (for example, the Wolfram user Applications directory)")

if(DiFfRG_MATHEMATICA_INSTALL_DIR)
  message(
    STATUS
      "DiFfRG Mathematica package install destination: ${DiFfRG_MATHEMATICA_INSTALL_DIR}")
  install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/Mathematica/DiFfRG
    MESSAGE_NEVER
    DESTINATION ${DiFfRG_MATHEMATICA_INSTALL_DIR}
    FILES_MATCHING
    PATTERN "*.m"
    PATTERN "*.wl"
    PATTERN "*.mx"
    PATTERN "*.nb")
else()
  message(
    STATUS
      "DiFfRG Mathematica package installation disabled; set "
      "DiFfRG_MATHEMATICA_INSTALL_DIR to enable it")
endif()
