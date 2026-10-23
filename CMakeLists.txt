# This file is part of GGEMS.
#
# GGEMS is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# GGEMS is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# GGEMS.  If not, see <https://www.gnu.org/licenses/>.

# ------------------------------------------------------------------------------
# CMakeLists.txt
#
# CMakeLists.txt - Compile and build the GGEMS library
#
# Authors : - Julien Bert <julien.bert@univ-brest.fr> - Didier Benoit
# <didier.benoit@inserm.fr>
#
# Generated on : 23/09/2019
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# Checking CMAKE version
cmake_minimum_required(VERSION 3.17 FATAL_ERROR)

# ------------------------------------------------------------------------------
# Setting the compiler type: 'CLANG', 'GCC', 'CL' (visual c++) or 'INTEL' Visual
# Studio: CL (only on Windows) and by default on Windows CC=cl CXX=cl GNU GCC:
# GCC (only on Unix) and by default on Unix CC=gcc CXX=g++ LLVM CLANG: CLANG
# (Validated on Unix and Windows) CC=clang CXX=clang++ INTEL (based on LLVM):
# INTEL (Validated on Unix and Windows) CC=icx CXX=icx (Windows) icpx (Unix)

if(WIN32)
  set(COMPILER
      CL
      CACHE INTERNAL "Compiler name: CLANG, CL, INTEL")
else()
  set(COMPILER
      GCC
      CACHE INTERNAL "Compiler name: CLANG, GCC, INTEL")
endif()

if(COMPILER STREQUAL "CLANG")
  set(ENV{CC} "clang")
  set(ENV{CXX} "clang++")
elseif(COMPILER STREQUAL "CL")
  set(ENV{CC} "cl")
  set(ENV{CXX} "cl")
elseif(COMPILER STREQUAL "GCC")
  set(ENV{CC} "gcc")
  set(ENV{CXX} "g++")
elseif(COMPILER STREQUAL "INTEL")
  set(ENV{CC} "icx")
  if(WIN32)
    set(ENV{CXX} "icx")
  else()
    set(ENV{CXX} "icpx")
  endif()
else()
  message(FATAL_ERROR "Compiler name unknown!!! Options are CLANG, GCC or CL")
endif()

# ------------------------------------------------------------------------------
# Name of the project and define the language
project(
  GGEMS
  VERSION 1.3
  HOMEPAGE_URL ggems.fr
  LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ------------------------------------------------------------------------------
# Give cmake-config directory to cmake
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake-config")

# ------------------------------------------------------------------------------
set(OPENCL_SDK_BUILD_UTILITY_LIBRARIES
    OFF
    CACHE BOOL "" FORCE)
set(OPENCL_SDK_BUILD_CLINFO
    OFF
    CACHE BOOL "" FORCE)
set(BUILD_TESTING
    OFF
    CACHE BOOL "" FORCE)
set(BUILD_EXAMPLES
    OFF
    CACHE BOOL "" FORCE)
set(BUILD_DOCS
    OFF
    CACHE BOOL "" FORCE)
set(ENABLE_OPENCL_LAYERINFO
    OFF
    CACHE BOOL "" FORCE)

add_subdirectory(external/OpenCL-SDK)

add_definitions("-DCL_TARGET_OPENCL_VERSION=300")
add_definitions("-DCL_HPP_TARGET_OPENCL_VERSION=300")

# ------------------------------------------------------------------------------
# Find libraries for OpenGL
option(OPENGL_VISUALIZATION "Using OpenGL for visualization" OFF)
if(OPENGL_VISUALIZATION)
  add_definitions(-DOPENGL_VISUALIZATION)
  find_package(GLFW3 REQUIRED)
  find_package(OpenGL REQUIRED)
  set(GLEW_USE_STATIC_LIBS TRUE) # Using GLEW in static, important for Python!!!
  find_package(GLEW REQUIRED)
  find_package(glm CONFIG REQUIRED)
  # GET_FILENAME_COMPONENT(GLM_INCLUDE_DIR ${glm_DIR}/../../include ABSOLUTE)
endif()

# ------------------------------------------------------------------------------
# Force the build type to Release
set(CMAKE_BUILD_TYPE
    "Release"
    CACHE STRING "Choose the type of build, options are: Debug Release" FORCE)
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release")

# ------------------------------------------------------------------------------
# Setting the LOGO_PATH variable
set(LOGO_PATH
    ${PROJECT_SOURCE_DIR}/logo
    CACHE PATH "Path to the logo repository")

# ------------------------------------------------------------------------------
# Setting the OPENCL_KERNEL_PATH variable
set(OPENCL_KERNEL_PATH
    ${PROJECT_SOURCE_DIR}/src/kernels
    CACHE PATH "Path to the OpenCL kernel repository")

# ------------------------------------------------------------------------------
# Setting the GGEMSHOME_PATH variable
set(GGEMS_PATH
    ${PROJECT_SOURCE_DIR}
    CACHE PATH "Path to the GGEMS project repository")

# ------------------------------------------------------------------------------
# Setting the maximum particles in OpenCL buffer
if(DEFINED MAXIMUM_PARTICLES)
  set(MAXIMUM_PARTICLES
      ${MAXIMUM_PARTICLES}
      CACHE STRING "Number of particles in OpenCL buffer")
else()
  set(MAXIMUM_PARTICLES
      1048576
      CACHE STRING "Number of particles in OpenCL buffer") # Validated on old
                                                           # graphic card as GTX
                                                           # 980 Ti
endif()

# ------------------------------------------------------------------------------
# Add an option for using cache kernel compilation on OpenCL device Set to OFF
# to be sure your own kernel modification are re-compiled
option(OPENCL_CACHE_KERNEL_COMPILATION
       "Using kernel cache compilation (NVIDIA only)" ON)
if(OPENCL_CACHE_KERNEL_COMPILATION)
  add_definitions(-DOPENCL_CACHE_KERNEL_COMPILATION)
endif()

# ------------------------------------------------------------------------------
# Add an option for using cache kernel compilation on OpenCL device Set to OFF
# to be sure your own kernel modification are re-compiled
option(DOSIMETRY_DOUBLE_PRECISION "Double precision for dosimetry" ON)
if(DOSIMETRY_DOUBLE_PRECISION)
  add_definitions(-DDOSIMETRY_DOUBLE_PRECISION)
endif()

# ------------------------------------------------------------------------------
# Defining a configuration file
configure_file(
  "${PROJECT_SOURCE_DIR}/cmake-config/GGEMSConfiguration.hh.in"
  "${PROJECT_SOURCE_DIR}/include/GGEMS/global/GGEMSConfiguration.hh" @ONLY)

# ------------------------------------------------------------------------------
# Compilation options Windows
if(WIN32)
  # Checking compiler for LLVM/CLANG
  if(COMPILER STREQUAL "CLANG")
    # Delete some flags by default provided by CMAKE
    string(REPLACE "-O3" "" CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE})
    string(REPLACE "-DNDEBUG" "" CMAKE_CXX_FLAGS_RELEASE
                   ${CMAKE_CXX_FLAGS_RELEASE})

    set(CXX_FLAGS
        -std=c++17
        -m64
        -march=native
        -fno-common
        -Wextra
        -Wall
        -Wundef
        -Wdouble-promotion
        -Wshadow
        -Wno-unused-parameter
        -Wconversion)
    set(CXX_FLAGS_RELEASE -O3 ${CXX_FLAGS})
    set(CXX_FLAGS_DEBUG -O0 -g3 ${CXX_FLAGS})

    # Checking compiler for Visual Studio
  elseif(COMPILER STREQUAL "CL")
    string(REPLACE "/W3" "" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
    string(REPLACE "/O2" "" CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE})
    string(REPLACE "/D NDEBUG" "" CMAKE_CXX_FLAGS_RELEASE
                   ${CMAKE_CXX_FLAGS_RELEASE})

    set(CXX_FLAGS /EHsc /nologo /W3 /std:c++17
                  /D_DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR)
    set(CXX_FLAGS_RELEASE /O2 /Ox ${CXX_FLAGS})
    set(CXX_FLAGS_DEBUG /Od /DEBUG /Zi ${CXX_FLAGS})

    # Checking compiler for Intel OneAPI
  elseif(COMPILER STREQUAL "INTEL")
    string(REPLACE "/W3" "" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
    string(REPLACE "/O2" "" CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE})
    string(REPLACE "/D NDEBUG" "" CMAKE_CXX_FLAGS_RELEASE
                   ${CMAKE_CXX_FLAGS_RELEASE})

    set(CXX_FLAGS
        /EHsc
        /O3
        /std:c++17
        /W3
        /Qm64
        /external:I${PROJECT_SOURCE_DIR}/include/externs
        /external:I${GLFW3_INCLUDE_DIR}
        /external:I${GLEW_INCLUDE_DIR}
        /external:I${OpenCL_INCLUDE_DIRS}
        /external:I${GLM_INCLUDE_DIR}
        -Wno-deprecated-declarations
        -Xclang
        -Wdouble-promotion
        -Wshadow
        -Wno-unused-parameter
        -Wconversion)
    set(CXX_FLAGS_RELEASE /O3 ${CXX_FLAGS})
    set(CXX_FLAGS_DEBUG /Od /DEBUG /Zi ${CXX_FLAGS})

    # Otherwize error
  else()
    message(FATAL_ERROR "Only 'CL' and 'CLANG' are available for Windows")
  endif()

  add_compile_options("$<$<CONFIG:RELEASE>:${CXX_FLAGS_RELEASE}>"
                      "$<$<CONFIG:DEBUG>:${CXX_FLAGS_DEBUG}>")

  # Compilation options Unix
else()
  # Finding CCACHE for faster compilation in unix
  find_program(CCACHE_FOUND ccache)
  if(CCACHE_FOUND)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
  endif(CCACHE_FOUND)

  # Delete some flags by default provided by CMAKE
  string(REPLACE "-O3" "" CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE})
  string(REPLACE "-DNDEBUG" "" CMAKE_CXX_FLAGS_RELEASE
                 ${CMAKE_CXX_FLAGS_RELEASE})

  # Checking compiler for LLVM/CLANG
  if(COMPILER STREQUAL "CLANG")
    set(CXX_FLAGS
        -std=c++17
        -m64
        -march=native
        -Wundef
        -Wdouble-promotion
        -Wshadow
        -Wconversion
        -fno-common
        -Wextra
        -Wall
        -pedantic
        -Wold-style-cast
        -pedantic-errors
        -Wno-unused-parameter)

    # Checking compiler for GNU/GCC
  elseif(COMPILER STREQUAL "GCC")
    set(CXX_FLAGS
        -std=c++17
        -m64
        -march=native
        -Wundef
        -Wdouble-promotion
        -Wshadow
        -Wconversion
        -fno-common
        -Wextra
        -Wall
        -pedantic
        -Wold-style-cast
        -pedantic-errors
        -Wno-unused-parameter)

    # Checking compiler for INTEL
  elseif(COMPILER STREQUAL "INTEL")
    set(CXX_FLAGS
        -std=c++17
        -m64
        -march=native
        -Wundef
        -Wdouble-promotion
        -Wshadow
        -Wconversion
        -fno-common
        -Wextra
        -Wall
        -pedantic
        -Wold-style-cast
        -pedantic-errors
        -Wno-unused-parameter)

    # Otherwize error
  else()
    message(FATAL_ERROR "Only 'CLANG' and 'GCC' are available for Unix")
  endif()

  set(CXX_FLAGS_RELEASE -O3 ${CXX_FLAGS})
  set(CXX_FLAGS_DEBUG -O0 -g3 ${CXX_FLAGS})

  add_compile_options("$<$<CONFIG:RELEASE>:${CXX_FLAGS_RELEASE}>"
                      "$<$<CONFIG:DEBUG>:${CXX_FLAGS_DEBUG}>")
endif()

# ------------------------------------------------------------------------------
# Setup include directory for GGEMS
include_directories(${PROJECT_SOURCE_DIR}/include)

# ------------------------------------------------------------------------------
# Locate source files for GGEMS
file(
  GLOB
  source_ggems
  ${PROJECT_SOURCE_DIR}/src/global/*.cc
  ${PROJECT_SOURCE_DIR}/src/geometries/*.cc
  ${PROJECT_SOURCE_DIR}/src/tools/*.cc
  ${PROJECT_SOURCE_DIR}/src/navigators/*.cc
  ${PROJECT_SOURCE_DIR}/src/physics/*.cc
  ${PROJECT_SOURCE_DIR}/src/sources/*.cc
  ${PROJECT_SOURCE_DIR}/src/randoms/*.cc
  ${PROJECT_SOURCE_DIR}/src/maths/*.cc
  ${PROJECT_SOURCE_DIR}/src/materials/*.cc
  ${PROJECT_SOURCE_DIR}/src/io/*.cc
  ${PROJECT_SOURCE_DIR}/src/graphics/*.cc)

# ------------------------------------------------------------------------------
# Export Header for DLL Windows
include(GenerateExportHeader)

# ------------------------------------------------------------------------------
# Create shared library
add_library(ggems SHARED ${source_ggems})
if(OPENGL_VISUALIZATION)
  target_link_libraries(
    ggems
    OpenCL::OpenCL
    OpenCL::HeadersCpp
    ${GLFW3_LIBRARY}
    OpenGL::GL
    OpenGL::GLU
    GLEW::glew_s
    glm::glm)
else()
  target_link_libraries(ggems OpenCL::OpenCL OpenCL::HeadersCpp)
endif()
set_target_properties(ggems PROPERTIES PREFIX "lib")

# ------------------------------------------------------------------------------
# DLL export for windows
generate_export_header(
  ggems EXPORT_FILE_NAME
  ${PROJECT_SOURCE_DIR}/include/GGEMS/global/GGEMSExport.hh)

# ------------------------------------------------------------------------------
# Installing GGEMS library
install(TARGETS ggems DESTINATION ggems/lib)
install(DIRECTORY include/GGEMS DESTINATION ggems/include)
