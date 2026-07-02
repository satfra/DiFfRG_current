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
# Install the Mathematica package (optional)
# ##############################################################################
#
# The Mathematica/Wolfram Language package is an OPTIONAL component: it is only
# used for symbolic derivation and C++ code generation of flow equations, which
# is irrelevant to building and using the compiled library. Machines without a
# Wolfram installation MUST still configure and install cleanly.
#
# Detection is a cheap, network-free find_program() probe. Everything the
# install needs — a Wolfram binary that can run get_wolfram_app_dir.m to report
# the user application directory ($UserBaseDirectory/Applications) — is obtained
# directly from that binary. We deliberately do NOT download CPM.cmake or
# WolframResearch/LibraryLinkUtilities (previously pulled in only for its
# FindWolframLanguage module): that made the install require network access even
# on machines that have Wolfram, and fail outright on machines that do not.

# Honor explicit user hints (cache or env) before probing PATH / default dirs.
set(_WOLFRAM_HINTS "")
if(DEFINED WolframLanguage_ROOT)
  list(APPEND _WOLFRAM_HINTS "${WolframLanguage_ROOT}")
endif()
if(DEFINED WolframLanguage_INSTALL_DIR)
  list(APPEND _WOLFRAM_HINTS "${WolframLanguage_INSTALL_DIR}")
endif()
if(DEFINED ENV{MATHEMATICA_HOME})
  list(APPEND _WOLFRAM_HINTS "$ENV{MATHEMATICA_HOME}")
endif()

# wolframscript is preferred (it is the canonical script runner), but a bare
# kernel (wolfram / WolframKernel / math) also accepts "-script" and is enough.
find_program(
  WOLFRAM_EXE
  NAMES wolframscript wolfram WolframKernel math MathKernel
  HINTS ${_WOLFRAM_HINTS}
  PATH_SUFFIXES Executables MacOS Contents/MacOS)

if(NOT WOLFRAM_EXE)
  message(
    STATUS
      "Wolfram Language / Mathematica not found — skipping the (optional) "
      "Mathematica package install. Symbolic flow-equation code generation will "
      "be unavailable; the compiled library is unaffected. Set "
      "WolframLanguage_ROOT or MATHEMATICA_HOME to enable it.")
  return()
endif()

message(STATUS "Wolfram binary detected: ${WOLFRAM_EXE}")

# Ask the interpreter for its user application directory. All candidate binaries
# accept "-script <file>". Capture the exit status so a broken/unlicensed
# install degrades to a warning instead of aborting the whole install.
execute_process(
  COMMAND ${WOLFRAM_EXE} -script
          ${CMAKE_CURRENT_SOURCE_DIR}/cmake/get_wolfram_app_dir.m
  OUTPUT_VARIABLE WOLFRAM_APP_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE WOLFRAM_APP_DIR_RESULT
  ERROR_QUIET)

if(NOT WOLFRAM_APP_DIR_RESULT EQUAL 0 OR WOLFRAM_APP_DIR STREQUAL "")
  message(
    WARNING
      "Wolfram binary '${WOLFRAM_EXE}' could not report its application "
      "directory (exit code ${WOLFRAM_APP_DIR_RESULT}). Skipping install of the "
      "Mathematica package.")
  return()
endif()

message(STATUS "Wolfram Language application directory: ${WOLFRAM_APP_DIR}")

# install the Mathematica package
install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/Mathematica/DiFfRG
  MESSAGE_NEVER
  DESTINATION ${WOLFRAM_APP_DIR}
  FILES_MATCHING
  PATTERN "*.m"
  PATTERN "*.wl"
  PATTERN "*.mx"
  PATTERN "*.nb")
