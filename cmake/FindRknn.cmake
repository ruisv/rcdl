# Locates the Rockchip RKNPU2 runtime (librknnrt + rknn_api.h) for an RK35xx
# board build.
#
# Sources, searched in this order:
#   1. The active conda env ($PREFIX / $BUILD_PREFIX / $CONDA_PREFIX) — when the
#      `librknnrt` conda package is installed (headers in include/, lib in lib/).
#   2. An explicit RKNN_ROOT (e.g. a checkout of the rknn-toolkit2 repo:
#      rknpu2/runtime/Linux/librknn_api).
#   3. The repo-local mirror third_party/rknpu2 (gitignored; populated by
#      scripts/fetch_sdk.sh) for the headers, and the board image for the lib
#      (/usr/lib/librknnrt.so, where every Rockchip Ubuntu/Debian image puts it).
#
# Provides the imported target
#   rknn::rt      (librknnrt.so + include dir)
# and variables RKNN_INCLUDE_DIR / RKNN_LIBRARY / RKNN_RUNTIME_LIB_DIR.

set(RKNN_ROOT "" CACHE PATH "Root of an rknpu2 runtime tree (contains include/rknn_api.h, lib or aarch64/librknnrt.so)")

set(_rknn_inc_hints "")
set(_rknn_lib_hints "")
foreach(_env PREFIX BUILD_PREFIX CONDA_PREFIX)
  if(DEFINED ENV{${_env}})
    list(APPEND _rknn_inc_hints "$ENV{${_env}}/include")
    list(APPEND _rknn_lib_hints "$ENV{${_env}}/lib")
  endif()
endforeach()
if(RKNN_ROOT)
  list(APPEND _rknn_inc_hints "${RKNN_ROOT}/include")
  list(APPEND _rknn_lib_hints "${RKNN_ROOT}/lib" "${RKNN_ROOT}/aarch64")
endif()
list(APPEND _rknn_inc_hints "${CMAKE_CURRENT_LIST_DIR}/../third_party/rknpu2/include")
list(APPEND _rknn_lib_hints "${CMAKE_CURRENT_LIST_DIR}/../third_party/rknpu2/lib")

find_path(RKNN_INCLUDE_DIR
  NAMES rknn_api.h
  HINTS ${_rknn_inc_hints}
  PATHS /usr/include /usr/include/rknn /usr/local/include)

find_library(RKNN_LIBRARY
  NAMES rknnrt
  HINTS ${_rknn_lib_hints}
  PATHS /usr/lib /usr/lib/aarch64-linux-gnu /usr/local/lib)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Rknn
  REQUIRED_VARS RKNN_INCLUDE_DIR RKNN_LIBRARY)

if(Rknn_FOUND)
  get_filename_component(RKNN_RUNTIME_LIB_DIR "${RKNN_LIBRARY}" DIRECTORY)
  if(NOT TARGET rknn::rt)
    add_library(rknn::rt UNKNOWN IMPORTED)
    set_target_properties(rknn::rt PROPERTIES
      IMPORTED_LOCATION "${RKNN_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${RKNN_INCLUDE_DIR}")
  endif()
endif()
