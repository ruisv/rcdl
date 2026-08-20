# Locates the Rockchip media stack used by RCDL's preproc + media layers:
#
#   RGA  (librga, im2d API)       -> 2-D hardware: resize / cvtColor / letterbox
#   MPP  (librockchip_mpp, rk_mpi) -> VPU: H.264 / H.265 / JPEG decode + encode
#
# Both are optional: RCDL builds without them (NPU inference + CPU preproc only)
# and guards the hardware paths on RCDL_HAVE_RGA / RCDL_HAVE_MPP.
#
# Sources, searched in this order: the active conda env (librga / rockchip-mpp
# packages), then the board image (Rockchip's librga-dev / librockchip-mpp-dev
# packages: /usr/include/rga, /usr/include/rockchip, /usr/lib/aarch64-linux-gnu),
# then the repo-local sysroot mirror third_party/sysroot (gitignored;
# scripts/fetch_sysroot.sh) — that last one is for IntelliSense on a
# workstation, not for linking.
#
# Provides imported targets (when found):
#   rockchip::rga   rockchip::mpp
# and RGA_FOUND / MPP_FOUND / RGA_INCLUDE_DIR / MPP_INCLUDE_DIR.

set(_rk_inc_hints "")
set(_rk_lib_hints "")
foreach(_env PREFIX BUILD_PREFIX CONDA_PREFIX)
  if(DEFINED ENV{${_env}})
    list(APPEND _rk_inc_hints "$ENV{${_env}}/include")
    list(APPEND _rk_lib_hints "$ENV{${_env}}/lib")
  endif()
endforeach()
set(_rk_sysroot "${CMAKE_CURRENT_LIST_DIR}/../third_party/sysroot")
list(APPEND _rk_inc_hints "${_rk_sysroot}/usr/include")
list(APPEND _rk_lib_hints "${_rk_sysroot}/usr/lib/aarch64-linux-gnu" "${_rk_sysroot}/usr/lib")

# --- RGA ----------------------------------------------------------------------
find_path(RGA_INCLUDE_DIR
  NAMES rga/im2d.h
  HINTS ${_rk_inc_hints}
  PATHS /usr/include /usr/local/include)
find_library(RGA_LIBRARY
  NAMES rga
  HINTS ${_rk_lib_hints}
  PATHS /usr/lib/aarch64-linux-gnu /usr/lib /usr/local/lib)

# --- MPP ----------------------------------------------------------------------
find_path(MPP_INCLUDE_DIR
  NAMES rockchip/rk_mpi.h
  HINTS ${_rk_inc_hints}
  PATHS /usr/include /usr/local/include)
find_library(MPP_LIBRARY
  NAMES rockchip_mpp
  HINTS ${_rk_lib_hints}
  PATHS /usr/lib/aarch64-linux-gnu /usr/lib /usr/local/lib)

include(FindPackageHandleStandardArgs)
# Both components are optional individually; the package "is found" if either is.
set(RGA_FOUND FALSE)
set(MPP_FOUND FALSE)
if(RGA_INCLUDE_DIR AND RGA_LIBRARY)
  set(RGA_FOUND TRUE)
  if(NOT TARGET rockchip::rga)
    add_library(rockchip::rga UNKNOWN IMPORTED)
    set_target_properties(rockchip::rga PROPERTIES
      IMPORTED_LOCATION "${RGA_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${RGA_INCLUDE_DIR}")
  endif()
endif()
if(MPP_INCLUDE_DIR AND MPP_LIBRARY)
  set(MPP_FOUND TRUE)
  if(NOT TARGET rockchip::mpp)
    add_library(rockchip::mpp UNKNOWN IMPORTED)
    set_target_properties(rockchip::mpp PROPERTIES
      IMPORTED_LOCATION "${MPP_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${MPP_INCLUDE_DIR}")
  endif()
endif()
if(RGA_FOUND OR MPP_FOUND)
  set(RockchipMedia_FOUND TRUE)
else()
  set(RockchipMedia_FOUND FALSE)
endif()
if(NOT RockchipMedia_FIND_QUIETLY)
  message(STATUS "Rockchip media: RGA=${RGA_FOUND} (${RGA_LIBRARY}) MPP=${MPP_FOUND} (${MPP_LIBRARY})")
endif()
