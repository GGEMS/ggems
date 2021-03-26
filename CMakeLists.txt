# ************************************************************************
# * This file is part of GGEMS.                                          *
# *                                                                      *
# * GGEMS is free software: you can redistribute it and/or modify        *
# * it under the terms of the GNU General Public License as published by *
# * the Free Software Foundation, either version 3 of the License, or    *
# * (at your option) any later version.                                  *
# *                                                                      *
# * GGEMS is distributed in the hope that it will be useful,             *
# * but WITHOUT ANY WARRANTY; without even the implied warranty of       *
# * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the        *
# * GNU General Public License for more details.                         *
# *                                                                      *
# * You should have received a copy of the GNU General Public License    *
# * along with GGEMS.  If not, see <https://www.gnu.org/licenses/>.      *
# *                                                                      *
# ************************************************************************

#-------------------------------------------------------------------------------
# CMakeLists.txt
#
# CMakeLists.txt - Compile and build the GGEMS library
#
# Authors :
#   - Julien Bert <julien.bert@univ-brest.fr>
#   - Didier Benoit <didier.benoit@inserm.fr>
#
# Generated on : 23/09/2019
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Checking CMAKE version
CMAKE_MINIMUM_REQUIRED(VERSION 3.8 FATAL_ERROR)

#-------------------------------------------------------------------------------
# Setting the compiler type: 'CLANG' 'GCC' or 'CL'
# Visual Studio: CL (Only on Windows) and by default on Windows
# CC=cl
# CXX=cl
# GNU GCC: GCC (Only validated on Unix)
# CC=gcc
# CXX=g++
# LLVM CLANG: CLANG (Validated on Unix and Windows) and by default on Windows
# CC=clang
# CXX=clang++
IF(WIN32)
  SET(COMPILER CL CACHE INTERNAL "Compiler name: CLANG, CL")
ELSE()
  SET(COMPILER GCC CACHE INTERNAL "Compiler name: CLANG, GCC")
ENDIF()

IF(COMPILER STREQUAL "CLANG")
  SET(ENV{CC} "clang")
  SET(ENV{CXX} "clang++")
ELSEIF(COMPILER STREQUAL "CL")
  SET(ENV{CC} "cl")
  SET(ENV{CXX} "cl")
ELSEIF(COMPILER STREQUAL "GCC")
  SET(ENV{CC} "gcc")
  SET(ENV{CXX} "g++")
ELSE()
  MESSAGE(FATAL_ERROR "Compiler name unknown!!! Options are CLANG, GCC or CL")
ENDIF()

#-------------------------------------------------------------------------------
# Name of the project and define the language
PROJECT(GGEMS VERSION 1.0 HOMEPAGE_URL ggems.fr LANGUAGES CXX)

#-------------------------------------------------------------------------------
# Find the OpenCL library
FIND_PACKAGE(OpenCL REQUIRED)

#-------------------------------------------------------------------------------
# Force the build type to Release
SET(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: Debug Release" FORCE)
SET_PROPERTY(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release")

#-------------------------------------------------------------------------------
# Setting the OPENCL_KERNEL_PATH variable
SET(OPENCL_KERNEL_PATH ${PROJECT_SOURCE_DIR}/src/kernels CACHE PATH "Path to the OpenCL kernel repository")

#-------------------------------------------------------------------------------
# Setting the GGEMSHOME_PATH variable
SET(GGEMS_PATH ${PROJECT_SOURCE_DIR} CACHE PATH "Path to the GGEMS project repository")

#-------------------------------------------------------------------------------
# Setting the PYTHON_MODULE_PATH variable
SET(PYTHON_MODULE_PATH ${PROJECT_SOURCE_DIR}/python_module CACHE PATH "Path to the python module repository")

#-------------------------------------------------------------------------------
# Setting the maximum particles in OpenCL buffer
IF(DEFINED MAXIMUM_PARTICLES)
  SET(MAXIMUM_PARTICLES ${MAXIMUM_PARTICLES} CACHE STRING "Number of particles in OpenCL buffer")
ELSE()
  SET(MAXIMUM_PARTICLES 1048576 CACHE STRING "Number of particles in OpenCL buffer") # Validated on old graphic card as GTX 980 Ti
ENDIF()

#-------------------------------------------------------------------------------
# Add an option for using cache kernel compilation on OpenCL device
# Set to OFF to be sure your own kernel modification are re-compiled
OPTION(OPENCL_CACHE_KERNEL_COMPILATION "Using kernel cache compilation" ON)
IF(OPENCL_CACHE_KERNEL_COMPILATION)
  ADD_DEFINITIONS(-DOPENCL_CACHE_KERNEL_COMPILATION)
ENDIF()

#-------------------------------------------------------------------------------
# Add an option for using cache kernel compilation on OpenCL device
# Set to OFF to be sure your own kernel modification are re-compiled
OPTION(DOSIMETRY_DOUBLE_PRECISION "Double precision for dosimetry" ON)
IF(DOSIMETRY_DOUBLE_PRECISION)
  ADD_DEFINITIONS(-DDOSIMETRY_DOUBLE_PRECISION)
ENDIF()

#-------------------------------------------------------------------------------
# Defining a configuration file
CONFIGURE_FILE("${PROJECT_SOURCE_DIR}/cmake-config/GGEMSConfiguration.hh.in" "${PROJECT_SOURCE_DIR}/include/GGEMS/global/GGEMSConfiguration.hh" @ONLY)

#-------------------------------------------------------------------------------
# Add an option for examples and benchmark building
OPTION(BUILD_EXAMPLES "Build GGEMS examples" ON)

#-------------------------------------------------------------------------------
# Compilation options Windows
IF(WIN32)
  # Checking compiler for LLVM/CLANG
  IF(COMPILER STREQUAL "CLANG")
    # Delete some flags by default provided by CMAKE
    STRING(REPLACE "-O3" "" CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE})
    STRING(REPLACE "-DNDEBUG" "" CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE})

    SET(CXX_FLAGS -std=c++17 -Wextra -Wall -pedantic -m64 -Wconversion -Weffc++ -Wold-style-cast -pedantic-errors)
    SET(CXX_FLAGS_RELEASE -O3 ${CXX_FLAGS})
    SET(CXX_FLAGS_DEBUG -O0 -g3 ${CXX_FLAGS})
  
  # Checking compiler for Visual Studio
  ELSEIF(COMPILER STREQUAL "CL")
    STRING(REPLACE "/W3" "" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
    STRING(REPLACE "/O2" "" CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE})
    STRING(REPLACE "/D NDEBUG" "" CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE})

    SET(CXX_FLAGS /MP /EHsc /nologo /W3 /std:c++17)
    SET(CXX_FLAGS_RELEASE /Ox ${CXX_FLAGS})
    SET(CXX_FLAGS_DEBUG /Od /DEBUG /Zi ${CXX_FLAGS})

    ELSEIF(COMPILER STREQUAL "GCC")
    STRING(REPLACE "-O3" "" CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE})
    STRING(REPLACE "-DNDEBUG" "" CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE})

    SET(CXX_FLAGS -std=c++17 -std=gnu++17 -Wextra -Wall -pedantic -m64 -Wconversion -Wold-style-cast -pedantic-errors)
    SET(CXX_FLAGS_RELEASE -O3 ${CXX_FLAGS})
    SET(CXX_FLAGS_DEBUG -O0 -g3 ${CXX_FLAGS})

  # Otherwize error
  ELSE()
    MESSAGE(FATAL_ERROR "Only 'CL' and 'CLANG' are available for Windows")
  ENDIF()

  ADD_COMPILE_OPTIONS("$<$<CONFIG:RELEASE>:${CXX_FLAGS_RELEASE}>" "$<$<CONFIG:DEBUG>:${CXX_FLAGS_DEBUG}>")

# Compilation options Unix
ELSE()
  # Finding CCACHE for faster compilation in unix
  FIND_PROGRAM(CCACHE_FOUND ccache)
  IF(CCACHE_FOUND)
    SET_PROPERTY(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
    SET_PROPERTY(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
  ENDIF(CCACHE_FOUND)

  # Delete some flags by default provided by CMAKE
  STRING(REPLACE "-O3" "" CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE})
  STRING(REPLACE "-DNDEBUG" "" CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE})

  # Checking compiler for LLVM/CLANG
  IF(COMPILER STREQUAL "CLANG")
    SET(CXX_FLAGS -std=c++17 -std=gnu++17 -Wall -Werror -Wextra -pedantic -m64 -Wconversion -Wold-style-cast -pedantic-errors)

  # Checking compiler for GNU/GCC
  ELSEIF(COMPILER STREQUAL "GCC")
    SET(CXX_FLAGS -std=c++17 -std=gnu++17 -Wall -pedantic -m64 -Wconversion -Wold-style-cast -pedantic-errors)

  # Otherwize error
  ELSE()
    MESSAGE(FATAL_ERROR "Only 'CLANG' and 'GCC' are available for Unix")
  ENDIF()

  SET(CXX_FLAGS_RELEASE -O3 ${CXX_FLAGS})
  SET(CXX_FLAGS_DEBUG -O0 -g3 ${CXX_FLAGS})

  ADD_COMPILE_OPTIONS("$<$<CONFIG:RELEASE>:${CXX_FLAGS_RELEASE}>" "$<$<CONFIG:DEBUG>:${CXX_FLAGS_DEBUG}>")
ENDIF()

#-------------------------------------------------------------------------------
# Setup include directory for GGEMS
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/include)
INCLUDE_DIRECTORIES(SYSTEM ${OpenCL_INCLUDE_DIRS})

#-------------------------------------------------------------------------------
# Add link directory for OpenCL
LINK_DIRECTORIES(${OpenCL_LIBRARY})

#-------------------------------------------------------------------------------
# Locate source files for GGEMS
FILE(GLOB source_ggems
  # ${PROJECT_SOURCE_DIR}/src/detectors/*.cc
  ${PROJECT_SOURCE_DIR}/src/global/*.cc
  # ${PROJECT_SOURCE_DIR}/src/geometries/*.cc
  ${PROJECT_SOURCE_DIR}/src/tools/*.cc
  # ${PROJECT_SOURCE_DIR}/src/navigators/*.cc
  # ${PROJECT_SOURCE_DIR}/src/physics/*.cc
  # ${PROJECT_SOURCE_DIR}/src/sources/*.cc
  # ${PROJECT_SOURCE_DIR}/src/randoms/*.cc
  # ${PROJECT_SOURCE_DIR}/src/maths/*.cc
  # ${PROJECT_SOURCE_DIR}/src/materials/*.cc
  # ${PROJECT_SOURCE_DIR}/src/io/*.cc
)

#-------------------------------------------------------------------------------
# Export Header for DLL Windows
INCLUDE(GenerateExportHeader)

#-------------------------------------------------------------------------------
# Create shared library
ADD_LIBRARY(ggems SHARED ${source_ggems})
TARGET_LINK_LIBRARIES(ggems ${OpenCL_LIBRARY})
SET_TARGET_PROPERTIES(ggems PROPERTIES PREFIX "lib")

#-------------------------------------------------------------------------------
# DLL export for windows
GENERATE_EXPORT_HEADER(ggems EXPORT_FILE_NAME ${PROJECT_SOURCE_DIR}/include/GGEMS/global/GGEMSExport.hh)

#------------------------------------------------------------------------------
# Building examples
IF(BUILD_EXAMPLES )
  ADD_SUBDIRECTORY(examples)
ENDIF()

#-------------------------------------------------------------------------------
# Installing GGEMS library
INSTALL(TARGETS ggems DESTINATION ggems/lib)
INSTALL(DIRECTORY include/GGEMS DESTINATION ggems/include)
INSTALL(DIRECTORY python_module DESTINATION ggems FILES_MATCHING PATTERN "*.py")
