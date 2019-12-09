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
# Name of the project and define the language
PROJECT(GGEMS VERSION 1.0 HOMEPAGE_URL ggems.fr LANGUAGES CXX)

#-------------------------------------------------------------------------------
# Find the OpenCL library
FIND_PACKAGE(OpenCL REQUIRED)

#-------------------------------------------------------------------------------
# Force the build type to Release
SET(CMAKE_BUILD_TYPE "Release" CACHE STRING
  "Choose the type of build, options are: Debug Release")
SET_PROPERTY(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release")

#-------------------------------------------------------------------------------
# Setting the compiler type: 'CLANG' 'GCC' or 'CL'
SET(COMPILER ${COMPILER})

#-------------------------------------------------------------------------------
# Setting the OPENCL_KERNEL_PATH variable
SET(OPENCL_KERNEL_PATH ${PROJECT_SOURCE_DIR}/src CACHE PATH
  "Path to the OpenCL kernel repository")

#-------------------------------------------------------------------------------
# Setting the GGEMSHOME_PATH variable
SET(GGEMS_PATH ${PROJECT_SOURCE_DIR} CACHE PATH
  "Path to the GGEMS project repository")

#-------------------------------------------------------------------------------
# Set a maximum of particle in a buffer
SET(MAXIMUM_PARTICLES "5000000"
  CACHE STRING "Maximum of particles simulated in a buffer")

#-------------------------------------------------------------------------------
# Defining a configuration file
CONFIGURE_FILE("${PROJECT_SOURCE_DIR}/cmake-config/ggems_configuration.hh.in"
  "${PROJECT_SOURCE_DIR}/include/GGEMS/global/ggems_configuration.hh"
  @ONLY)

#-------------------------------------------------------------------------------
# Compilation options Windows/Unix
IF(WIN32)
  # Checking compiler for Windows
  IF(COMPILER STREQUAL "GCC")
    # Delete some flags by default provided by CMAKE
    STRING(REPLACE "-O3" "" CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE})
    STRING(REPLACE "-DNDEBUG" "" CMAKE_CXX_FLAGS_RELEASE
      ${CMAKE_CXX_FLAGS_RELEASE})

    SET(CXX_FLAGS -std=c++17 -Wall -Wextra -pedantic -m64 -march=native
      -Wconversion -Wold-style-cast -pedantic-errors -ftree-vectorize
      -mtune=native -mavx2)
    SET(CXX_FLAGS_RELEASE -Ofast ${CXX_FLAGS})
    SET(CXX_FLAGS_DEBUG -O0 -g3 ${CXX_FLAGS})
  ELSEIF(COMPILER STREQUAL "CLANG")
    # Delete some flags by default provided by CMAKE
    STRING(REPLACE "-O3" "" CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE})
    STRING(REPLACE "-DNDEBUG" "" CMAKE_CXX_FLAGS_RELEASE
      ${CMAKE_CXX_FLAGS_RELEASE})

    SET(CXX_FLAGS -std=c++2a -Wall -mavx2 -Wextra -pedantic -m64
    -march=native -Wconversion -Weffc++ -Wold-style-cast
    -pedantic-errors -mtune=native -ftree-vectorize)

    SET(CXX_FLAGS_RELEASE -Ofast ${CXX_FLAGS})
    SET(CXX_FLAGS_DEBUG -O0 -g3 ${CXX_FLAGS})
  ELSEIF(COMPILER STREQUAL "CL")
    STRING(REPLACE "/W3" "" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
    STRING(REPLACE "/O2" "" CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE})
    STRING(REPLACE "/D NDEBUG" "" CMAKE_CXX_FLAGS_RELEASE
      ${CMAKE_CXX_FLAGS_RELEASE})

    SET(CXX_FLAGS /MP /EHsc /nologo /W3 /std:c++17)
    SET(CXX_FLAGS_RELEASE /Ox /Qpar ${CXX_FLAGS})
    SET(CXX_FLAGS_DEBUG /Od /DEBUG /Zi ${CXX_FLAGS})
  ELSE()
    MESSAGE(FATAL_ERROR
      "Only 'CL', 'CLANG' and 'GCC' are available for Windows")
  ENDIF()

  ADD_COMPILE_OPTIONS("$<$<CONFIG:RELEASE>:${CXX_FLAGS_RELEASE}>"
    "$<$<CONFIG:DEBUG>:${CXX_FLAGS_DEBUG}>")
ELSE()
  # Finding CCACHE for faster compilation in unix
  FIND_PROGRAM(CCACHE_FOUND ccache)
  IF(CCACHE_FOUND)
    SET_PROPERTY(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
    SET_PROPERTY(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
  ENDIF(CCACHE_FOUND)

  # Delete some flags by default provided by CMAKE
  STRING(REPLACE "-O3" "" CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE})
  STRING(REPLACE "-DNDEBUG" "" CMAKE_CXX_FLAGS_RELEASE
    ${CMAKE_CXX_FLAGS_RELEASE})

  # Checking compiler for Unix
  IF(COMPILER STREQUAL "CLANG")
    SET(CXX_FLAGS -std=c++2a -Wall -mavx2 -Wextra -pedantic -m64
      -march=native -Wconversion -Weffc++ -Wold-style-cast
      -pedantic-errors -mtune=native -ftree-vectorize)
  ELSEIF(COMPILER STREQUAL "GCC")
    SET(CXX_FLAGS -std=c++17 -Wall -Wextra -pedantic -m64 -march=native
      -Wconversion -Weffc++ -Wold-style-cast
      -pedantic-errors -ftree-vectorize -mtune=native -mavx2)
  ELSE()
    MESSAGE(FATAL_ERROR "Only 'CLANG' and 'GCC' are available for Unix")
  ENDIF()
  SET(CXX_FLAGS_RELEASE -Ofast ${CXX_FLAGS})
  SET(CXX_FLAGS_DEBUG -O0 -g3 ${CXX_FLAGS})

  ADD_COMPILE_OPTIONS("$<$<CONFIG:RELEASE>:${CXX_FLAGS_RELEASE}>"
    "$<$<CONFIG:DEBUG>:${CXX_FLAGS_DEBUG}>")
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
  ${PROJECT_SOURCE_DIR}/src/detectors/*.cc
  ${PROJECT_SOURCE_DIR}/src/global/*.cc
  ${PROJECT_SOURCE_DIR}/src/geometries/*.cc
  ${PROJECT_SOURCE_DIR}/src/tools/*.cc
  ${PROJECT_SOURCE_DIR}/src/navigators/*.cc
  ${PROJECT_SOURCE_DIR}/src/processes/*.cc
  ${PROJECT_SOURCE_DIR}/src/sources/*.cc
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
GENERATE_EXPORT_HEADER(ggems EXPORT_FILE_NAME 
  ${PROJECT_SOURCE_DIR}/include/GGEMS/global/ggems_export.hh)

#-------------------------------------------------------------------------------
# Installing GGEMS library
INSTALL(TARGETS ggems DESTINATION lib)
INSTALL(DIRECTORY include/GGEMS DESTINATION include)
