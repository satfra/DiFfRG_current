SCRIPT_PATH="$(
  cd -- "$(dirname "$0")" >/dev/null 2>&1
  pwd -P
)"
BUILD_PATH=${SCRIPT_PATH}/${LIBRARY_NAME}_build
INSTALL_PATH=${SCRIPT_PATH}/${LIBRARY_NAME}_install
SOURCE_PATH=${SCRIPT_PATH}/${LIBRARY_NAME}
CMAKE_LOG_FILE=${SCRIPT_PATH}/logs/${LIBRARY_NAME}_cmake_build.log
MAKE_LOG_FILE=${SCRIPT_PATH}/logs/${LIBRARY_NAME}_make_build.log
