if(NOT WIN32)
  string(ASCII 27 Esc)
  set(ColourReset "${Esc}[m")
  set(ColourBold "${Esc}[1m")
  set(Red "${Esc}[31m")
  set(Green "${Esc}[32m")
  set(Yellow "${Esc}[33m")
  set(Blue "${Esc}[34m")
  set(Magenta "${Esc}[35m")
  set(Cyan "${Esc}[36m")
  set(White "${Esc}[37m")
  set(BoldRed "${Esc}[1;31m")
  set(BoldGreen "${Esc}[1;32m")
  set(BoldYellow "${Esc}[1;33m")
  set(BoldBlue "${Esc}[1;34m")
  set(BoldMagenta "${Esc}[1;35m")
  set(BoldCyan "${Esc}[1;36m")
  set(BoldWhite "${Esc}[1;37m")
endif()

# ##############################################################################
# Native-architecture reporting
# ##############################################################################
#
# Announce whether the build targets the local CPU (-march=native) or a generic,
# portable baseline. When native, probe the C++ compiler's predefined macros to
# list which ISA extensions -march=native actually enables (AVX-512, AVX2, FMA,
# ...), so it is clear what the resulting binaries will require to run -- e.g. an
# AVX-512 build will SIGILL on a machine without AVX-512.
function(diffrg_report_native _native _cxx)
  if(NOT _native)
    message(
      "  ${BoldYellow}[NATIVE=OFF] Portable build: -march=native disabled; targeting a generic CPU.${ColourReset}"
    )
    return()
  endif()

  message(
    "  ${BoldGreen}[NATIVE=ON] Optimizing for the build machine's CPU (-march=native).${ColourReset}"
  )

  if(NOT _cxx)
    return()
  endif()
  # Dump the compiler's predefined macros under -march=native. Works for
  # GCC/Clang/AppleClang; on anything else (or if -march=native is rejected,
  # e.g. some Apple-silicon toolchains) we simply skip the capability list.
  execute_process(
    COMMAND ${_cxx} -march=native -dM -E -x c++ /dev/null
    OUTPUT_VARIABLE _macros
    ERROR_QUIET
    RESULT_VARIABLE _rv)
  if(NOT _rv EQUAL 0)
    message(
      "  ${BoldYellow}    (could not probe -march=native capabilities for this compiler)${ColourReset}"
    )
    return()
  endif()

  # Curated, highest-first list of ISA feature macros worth reporting.
  set(_features
      AVX512F AVX512DQ AVX512BW AVX512VL AVX512CD AVX512VNNI AVX512BF16 AVX2 AVX
      FMA F16C BMI BMI2 POPCNT SSE4_2 SSE4_1 SSSE3 SSE3 SSE2)
  set(_enabled "")
  foreach(_f ${_features})
    if(_macros MATCHES "#define __${_f}__ 1")
      string(REGEX REPLACE "^AVX512" "AVX-512" _name "${_f}")
      string(REGEX REPLACE "^SSE4_" "SSE4." _name "${_name}")
      list(APPEND _enabled "${_name}")
    endif()
  endforeach()

  if(_enabled)
    string(REPLACE ";" ", " _enabled_str "${_enabled}")
    message(
      "  ${BoldGreen}    Enabled CPU features: ${_enabled_str}${ColourReset}")
  endif()
endfunction()
