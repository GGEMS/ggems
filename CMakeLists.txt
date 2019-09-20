# Authors:
# Julien Bert julien.bert@univ-brest.fr
# Didier Benoit didier.benoit@inserm.fr
# Date: September 4th, 2019

#-------------------------------------------------------------------------------
# Define a minimal version of CMAKE
CMAKE_MINIMUM_REQUIRED(VERSION 3.8 FATAL_ERROR)

#-------------------------------------------------------------------------------
# Name of the project and define the language
PROJECT(GGEMS VERSION 2.1 HOMEPAGE_URL ggems.fr LANGUAGES CUDA CXX)

#------------------------------------------------------------------------------
# Force the build type to Release
SET(CMAKE_BUILD_TYPE "Release" CACHE STRING
  "Choose the type of build, options are: Debug Release")
SET_PROPERTY(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release")

#------------------------------------------------------------------------------
# Defining configuration files
CONFIGURE_FILE("${PROJECT_SOURCE_DIR}/cmake-config/ggems_configuration.hh.in"
  "${PROJECT_SOURCE_DIR}/global/ggems_configuration.hh"
  @ONLY)
                                                                               
#------------------------------------------------------------------------------
# Setting the GGEMSHOME_PATH variable
SET(GGEMSHOME ${CMAKE_INSTALL_PREFIX} CACHE PATH
  "Path to the GGEMS installation")

#-------------------------------------------------------------------------------
# On Unix system only, check if ccache is installed for faster compilation
FIND_PROGRAM(CCACHE_FOUND ccache)
IF(CCACHE_FOUND)
  SET_PROPERTY(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
  SET_PROPERTY(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
ENDIF(CCACHE_FOUND)

#-------------------------------------------------------------------------------
# Check if CUDA is installed on the computer and check the CUDA version
INCLUDE(CheckLanguage)
CHECK_LANGUAGE(CUDA)
IF(CMAKE_CUDA_COMPILER)
  ENABLE_LANGUAGE(CUDA)
  IF(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 7.0)
    MESSAGE(FATAL_ERROR "CUDA version should be >= 7.0")
  ENDIF()
ELSE()
  MESSAGE(FATAL_ERROR "No CUDA compiler found!!!")
ENDIF()

#-------------------------------------------------------------------------------
# Define CUDA flags by default
SET(CMAKE_CUDA_FLAGS "--machine 64 --std c++11")

#-------------------------------------------------------------------------------
# Compile GGEMS on the correct architecture
# https://en.wikipedia.org/wiki/CUDA
# Maxwell
OPTION(COMPUTE_CAPABILITY_50 "Maxwell architecture 5.0" OFF)
OPTION(COMPUTE_CAPABILITY_52 "Maxwell architecture 5.2" OFF)
IF(COMPUTE_CAPABILITY_50)
  STRING(APPEND CMAKE_CUDA_FLAGS " --generate-code arch=compute_50,code=sm_50")
ENDIF()
IF(COMPUTE_CAPABILITY_52)
  STRING(APPEND CMAKE_CUDA_FLAGS " --generate-code arch=compute_52,code=sm_52")
ENDIF()

# Pascal
OPTION(COMPUTE_CAPABILITY_60 "Pascal architecture 6.0" OFF)
OPTION(COMPUTE_CAPABILITY_61 "Pascal architecture 6.1" OFF)
IF(COMPUTE_CAPABILITY_60)
  STRING(APPEND CMAKE_CUDA_FLAGS " --generate-code arch=compute_60,code=sm_60")
ENDIF()
IF(COMPUTE_CAPABILITY_61)
  STRING(APPEND CMAKE_CUDA_FLAGS " --generate-code arch=compute_61,code=sm_61")
ENDIF()

# Volta
OPTION(COMPUTE_CAPABILITY_70 "Volta architecture 7.0" OFF)
IF(COMPUTE_CAPABILITY_70)
  STRING(APPEND CMAKE_CUDA_FLAGS " --generate-code arch=compute_70,code=sm_70")
ENDIF()

# Turing
OPTION(COMPUTE_CAPABILITY_75 "Turing architecture 7.5" OFF)
IF(COMPUTE_CAPABILITY_75)
  STRING(APPEND CMAKE_CUDA_FLAGS " --generate-code arch=compute_75,code=sm_75")
ENDIF()

#-------------------------------------------------------------------------------
# Activate the double precision for float
OPTION(DOUBLE_PRECISION "float number with double precision" OFF)
IF(NOT DOUBLE_PRECISION)
  ADD_DEFINITIONS(-DSINGLE_PRECISION)
ENDIF()

#-------------------------------------------------------------------------------
# Activate fast math compilation
OPTION(FAST_MATH "Fast math flag" OFF)
IF(FAST_MATH)
  STRING(APPEND CMAKE_CUDA_FLAGS " -use_fast_math")
ENDIF()

#-------------------------------------------------------------------------------
# List per kernel register, shared and constant memory usage
OPTION(KERNEL_MEMORY_INFO
  "List kernel register, shared and constant memory usage" OFF)
IF(KERNEL_MEMORY_INFO)
  STRING(APPEND CMAKE_CUDA_FLAGS " --ptxas-options=-v -Xptxas=-v")
ENDIF()

#------------------------------------------------------------------------------
# Setup include directory for GGEMS project
INCLUDE_DIRECTORIES(
  ${PROJECT_SOURCE_DIR}/alphadev
  ${PROJECT_SOURCE_DIR}/detectors
  ${PROJECT_SOURCE_DIR}/geometries
  ${PROJECT_SOURCE_DIR}/global
  ${PROJECT_SOURCE_DIR}/tools
  ${PROJECT_SOURCE_DIR}/navigators
  ${PROJECT_SOURCE_DIR}/processes
  ${PROJECT_SOURCE_DIR}/sources)

#------------------------------------------------------------------------------
# Locate sources for GGEMS
FILE(GLOB sources_ggems
  ${PROJECT_SOURCE_DIR}/alphadev/vox_phan_vrt_nav.cu
  ${PROJECT_SOURCE_DIR}/alphadev/vox_phan_ivrt_nav.cu
  ${PROJECT_SOURCE_DIR}/detectors/*.cu
  ${PROJECT_SOURCE_DIR}/geometries/*.cu
  ${PROJECT_SOURCE_DIR}/global/*.cu
  ${PROJECT_SOURCE_DIR}/tools/*.cu
  ${PROJECT_SOURCE_DIR}/navigators/*.cu
  ${PROJECT_SOURCE_DIR}/processes/*.cu
  ${PROJECT_SOURCE_DIR}/sources/*.cu)

#------------------------------------------------------------------------------
# Create shared GGEMS library
#ADD_LIBRARY(ggems SHARED ${sources_ggems})
ADD_LIBRARY(ggems STATIC ${sources_ggems})
SET_TARGET_PROPERTIES(ggems PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CUDA_SEPARABLE_COMPILATION ON)
TARGET_COMPILE_OPTIONS(ggems INTERFACE
  $<$<COMPILE_LANGUAGE:CUDA>:${CMAKE_CUDA_FLAGS}>)

#------------------------------------------------------------------------------
# Installing GGEMS library
INSTALL(TARGETS ggems DESTINATION lib)
INSTALL(DIRECTORY ${CMAKE_SOURCE_DIR}/alphadev/
  DESTINATION include/GGEMS FILES_MATCHING PATTERN "*.cuh*")
INSTALL(DIRECTORY ${CMAKE_SOURCE_DIR}/data/
  DESTINATION include/GGEMS FILES_MATCHING PATTERN "*.bin*")
INSTALL(DIRECTORY ${CMAKE_SOURCE_DIR}/detectors/
  DESTINATION include/GGEMS FILES_MATCHING PATTERN "*.cuh*")
INSTALL(DIRECTORY ${CMAKE_SOURCE_DIR}/geometries/
  DESTINATION include/GGEMS FILES_MATCHING PATTERN "*.cuh*")
INSTALL(DIRECTORY ${CMAKE_SOURCE_DIR}/global/
  DESTINATION include/GGEMS FILES_MATCHING PATTERN "*.cuh*")
INSTALL(DIRECTORY ${CMAKE_SOURCE_DIR}/global/
  DESTINATION include/GGEMS FILES_MATCHING PATTERN "*.hh*")
INSTALL(DIRECTORY ${CMAKE_SOURCE_DIR}/navigators/
  DESTINATION include/GGEMS FILES_MATCHING PATTERN "*.cuh*")
INSTALL(DIRECTORY ${CMAKE_SOURCE_DIR}/processes/
  DESTINATION include/GGEMS FILES_MATCHING PATTERN "*.cuh*")
INSTALL(DIRECTORY ${CMAKE_SOURCE_DIR}/sources/
  DESTINATION include/GGEMS FILES_MATCHING PATTERN "*.cuh*")
INSTALL(DIRECTORY ${CMAKE_SOURCE_DIR}/tools/
  DESTINATION include/GGEMS FILES_MATCHING PATTERN "*.cuh*")
INSTALL(DIRECTORY ${CMAKE_SOURCE_DIR}/data/
  DESTINATION include/GGEMS FILES_MATCHING PATTERN "*.bin*")
