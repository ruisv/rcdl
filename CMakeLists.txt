cmake_minimum_required(VERSION 3.18)
project(rcdl LANGUAGES CXX VERSION 0.1.0)

# --- Guard: this project only builds on an aarch64 Rockchip board -------------
# The RKNPU2 runtime, RGA and MPP are aarch64 userspace libraries that talk to
# kernel drivers (/dev/rknpu, /dev/rga, /dev/mpp_service). Edit anywhere; build
# and test on the board (scripts/sync.sh + scripts/board_build.sh).
if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
  message(FATAL_ERROR
    "RCDL builds on an aarch64 Rockchip board (RK3588 / RK3576 / RK356x) only. "
    "Current arch: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(GNUInstallDirs)

option(RCDL_BUILD_PYTHON "Build the nanobind Python module" ON)
option(RCDL_BUILD_EXAMPLES "Build example executables" ON)
option(RCDL_INSTALL "Generate install + find_package(rcdl) export rules" ON)
option(RCDL_WITH_RGA "Use the RGA 2-D engine for preprocessing when available" ON)
option(RCDL_WITH_MPP "Use MPP (VPU) for video/JPEG codecs when available" ON)

find_package(Rknn REQUIRED)
find_package(Threads REQUIRED)   # std::thread in the async pipelines
# OpenMP parallelizes the CPU preproc fallbacks. Optional.
find_package(OpenMP)
find_package(RockchipMedia QUIET)

# OpenCV (conda-forge libopencv in the board's `rcdl` env, or the image's
# libopencv-dev). Optional: code guards on RCDL_HAVE_OPENCV.
if(DEFINED ENV{CONDA_PREFIX})
  list(APPEND CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}")
  set(RCDL_CONDA_LIB "$ENV{CONDA_PREFIX}/lib")
endif()
find_package(OpenCV CONFIG QUIET)
if(OpenCV_FOUND)
  message(STATUS "OpenCV ${OpenCV_VERSION} found at ${OpenCV_DIR}")
else()
  message(STATUS "OpenCV not found — image helpers use hand-written fallbacks")
endif()

# Runtime rpath: where librknnrt was linked from (conda lib or /usr/lib) plus
# the conda lib dir when OpenCV comes from conda (it needs conda's libstdc++).
set(RCDL_RPATH "${RKNN_RUNTIME_LIB_DIR}")
if(OpenCV_FOUND AND DEFINED RCDL_CONDA_LIB)
  list(APPEND RCDL_RPATH "${RCDL_CONDA_LIB}")
  link_directories("${RCDL_CONDA_LIB}")
  add_link_options("-Wl,--disable-new-dtags")
endif()
list(REMOVE_DUPLICATES RCDL_RPATH)

# --- core library ------------------------------------------------------------
add_library(rcdl SHARED
  src/core/status.cc
  src/core/dma_buf.cc
  src/backend/engine.cc
  src/backend/output_reader.cc
  src/preproc/image.cc
  src/preproc/rga.cc
  src/preproc/letterbox_cpu.cc
  src/preproc/letterbox.cc
  src/tasks/detection.cc
  src/tasks/classification.cc
  src/tasks/embedding.cc
  src/tasks/pose.cc
  src/tasks/obb.cc
  src/tasks/instance_seg.cc
  src/tasks/segmentation.cc
  src/tasks/depth.cc
  src/tasks/ocr.cc
  src/tasks/face.cc
  src/tasks/features.cc
  src/media/mpp_common.cc
  src/media/video_frame.cc
  src/media/video_decoder.cc
  src/media/video_encoder.cc
  src/media/jpeg_codec.cc
  src/tracks/byte_tracker.cc
  src/tracks/reid.cc
  src/backend/engine_pool.cc
  src/pipeline/detection_pipeline.cc
  src/pipeline/async_detection_pipeline.cc
  src/pipeline/async_video_detection_pipeline.cc
  src/pipeline/tracking_pipeline.cc)
add_library(rcdl::rcdl ALIAS rcdl)

configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/version.h.in"
  "${CMAKE_CURRENT_BINARY_DIR}/include/rcdl/version.h" @ONLY)

target_include_directories(rcdl PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(rcdl PUBLIC rknn::rt Threads::Threads)
if(OpenMP_CXX_FOUND)
  target_link_libraries(rcdl PUBLIC OpenMP::OpenMP_CXX)
endif()

# --- RGA (2-D hardware preproc) ----------------------------------------------
set(RCDL_HAVE_RGA OFF)
if(RCDL_WITH_RGA AND RGA_FOUND)
  set(RCDL_HAVE_RGA ON)
  target_link_libraries(rcdl PUBLIC rockchip::rga)
  target_compile_definitions(rcdl PUBLIC RCDL_HAVE_RGA=1)
  message(STATUS "RGA hardware preproc: enabled")
else()
  message(STATUS "RGA hardware preproc: disabled (librga not found or RCDL_WITH_RGA=OFF)")
endif()

# --- MPP (VPU codecs) ---------------------------------------------------------
set(RCDL_HAVE_MPP OFF)
if(RCDL_WITH_MPP AND MPP_FOUND)
  set(RCDL_HAVE_MPP ON)
  target_link_libraries(rcdl PUBLIC rockchip::mpp)
  target_compile_definitions(rcdl PUBLIC RCDL_HAVE_MPP=1)
  message(STATUS "MPP video/JPEG codecs: enabled")
else()
  message(STATUS "MPP video/JPEG codecs: disabled (librockchip_mpp not found or RCDL_WITH_MPP=OFF)")
endif()

if(OpenCV_FOUND)
  target_link_libraries(rcdl PUBLIC ${OpenCV_LIBS})
  target_compile_definitions(rcdl PUBLIC RCDL_HAVE_OPENCV=1)
endif()
target_compile_options(rcdl PRIVATE -Wall -Wextra)
set_target_properties(rcdl PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  BUILD_RPATH "${RCDL_RPATH}" INSTALL_RPATH "${RCDL_RPATH}")

# --- examples ----------------------------------------------------------------
if(RCDL_BUILD_EXAMPLES)
  function(rcdl_example name)
    add_executable(${name} examples/${name}.cc)
    target_link_libraries(${name} PRIVATE rcdl::rcdl)
    set_target_properties(${name} PROPERTIES
      BUILD_RPATH "${RCDL_RPATH}" INSTALL_RPATH "${RCDL_RPATH}")
  endfunction()
  rcdl_example(model_info)
  rcdl_example(npu_bench)
  rcdl_example(dma_buf_probe)
  rcdl_example(det_demo)
  rcdl_example(video_decode)
  rcdl_example(video_roundtrip)
  rcdl_example(jpeg_roundtrip)
  rcdl_example(video_det_demo)
  rcdl_example(video_det_async)
  rcdl_example(async_bench)
  rcdl_example(cls_demo)
  rcdl_example(seg_demo)
  rcdl_example(depth_demo)
  rcdl_example(xfeat_demo)
  rcdl_example(async_check)
endif()

# --- python module -----------------------------------------------------------
if(RCDL_BUILD_PYTHON)
  find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT
    RESULT_VARIABLE _nb_rc)
  if(NOT _nb_rc EQUAL 0)
    message(FATAL_ERROR "nanobind not found in this env: pip/conda install nanobind "
                        "(or configure with -DRCDL_BUILD_PYTHON=OFF)")
  endif()
  find_package(nanobind CONFIG REQUIRED)
  nanobind_add_module(rcdl_py src/python/rcdl_py.cc)
  target_link_libraries(rcdl_py PRIVATE rcdl::rcdl)
  set_target_properties(rcdl_py PROPERTIES
    BUILD_RPATH "${RCDL_RPATH}"
    INSTALL_RPATH "$ORIGIN/lib;${RCDL_RPATH}")

  if(DEFINED SKBUILD)
    install(TARGETS rcdl_py LIBRARY DESTINATION ".")
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/python/rcdl_py.pyi")
      install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/python/rcdl_py.pyi" DESTINATION ".")
    endif()
  else()
    set(RCDL_SITE_DIR
      "${CMAKE_INSTALL_LIBDIR}/python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}/site-packages")
    install(TARGETS rcdl_py LIBRARY DESTINATION "${RCDL_SITE_DIR}")
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/python/rcdl_py.pyi")
      install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/python/rcdl_py.pyi" DESTINATION "${RCDL_SITE_DIR}")
    endif()
    install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/python/rcdl" DESTINATION "${RCDL_SITE_DIR}")
  endif()

  option(RCDL_BUILD_STUBS "Regenerate the rcdl_py.pyi type stub (imports the module)" OFF)
  if(RCDL_BUILD_STUBS)
    nanobind_add_stub(rcdl_py_stub
      MODULE rcdl_py
      OUTPUT "${CMAKE_CURRENT_SOURCE_DIR}/python/rcdl_py.pyi"
      PYTHON_PATH $<TARGET_FILE_DIR:rcdl_py>
      DEPENDS rcdl_py)
  endif()
endif()

# --- install / export: find_package(rcdl) for downstream consumers -----------
if(RCDL_INSTALL)
  include(CMakePackageConfigHelpers)

  install(TARGETS rcdl EXPORT rcdlTargets
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
    INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
  install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/rcdl"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
  install(FILES "${CMAKE_CURRENT_BINARY_DIR}/include/rcdl/version.h"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/rcdl")

  set(_cmake_dir "${CMAKE_INSTALL_LIBDIR}/cmake/rcdl")
  install(EXPORT rcdlTargets NAMESPACE rcdl:: DESTINATION "${_cmake_dir}")

  set(RCDL_PACKAGE_OPTIONAL_DEPS "")
  if(RCDL_HAVE_RGA OR RCDL_HAVE_MPP)
    string(APPEND RCDL_PACKAGE_OPTIONAL_DEPS "find_dependency(RockchipMedia)\n")
  endif()
  if(OpenMP_CXX_FOUND)
    string(APPEND RCDL_PACKAGE_OPTIONAL_DEPS "find_dependency(OpenMP)\n")
  endif()
  if(OpenCV_FOUND)
    string(APPEND RCDL_PACKAGE_OPTIONAL_DEPS "find_dependency(OpenCV CONFIG)\n")
  endif()
  configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/rcdlConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/rcdlConfig.cmake"
    INSTALL_DESTINATION "${_cmake_dir}")
  write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/rcdlConfigVersion.cmake"
    VERSION "${PROJECT_VERSION}" COMPATIBILITY SameMinorVersion)
  install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/rcdlConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/rcdlConfigVersion.cmake"
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindRknn.cmake"
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindRockchipMedia.cmake"
    DESTINATION "${_cmake_dir}")
endif()
