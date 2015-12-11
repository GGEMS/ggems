#----------------------------------------------------------------------------
# Setup the project
#
cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

SET( CMAKE_BUILD_TYPE Release )
IF( DEFINED CMAKE_BUILD_TYPE )
  SET( CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE} CACHE STRING "" )
ELSE()
  SET( CMAKE_BUILD_TYPE Debug CACHE STRING "" )
ENDIF()

project(GGEMS)

if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
    message(STATUS "GGEMS requires an out-of-source build.")
    message(STATUS "Please remove these files from ${CMAKE_BINARY_DIR} first:")
    message(STATUS "CMakeCache.txt")
    message(STATUS "CMakeFiles")
    message(STATUS "Once these files are removed, create a separate directory")
    message(STATUS "and run CMake from there")
    message(FATAL_ERROR "in-source build detected")
endif()

# OPTIONS
option(WITH_DOUBLE_PRECISION "Float numbers can be computed in double precision" ON)
if(WITH_DOUBLE_PRECISION)
  #add_definitions(-DSINGLE_PRECISION=0)
else()
  add_definitions(-DSINGLE_PRECISION)
endif()

# OPTIONS
option(WITH_FAST_MATH "Fast math flag" ON)
if(WITH_FAST_MATH)
set(LIBSCUDA --generate-code arch=compute_30,code=sm_30 --generate-code arch=compute_50,code=sm_50 --generate-code arch=compute_52,code=sm_52 --relocatable-device-code true -lcudadevrt --compiler-options -use_fast_math -w)

else()
  set(LIBSCUDA --generate-code arch=compute_30,code=sm_30 --generate-code arch=compute_50,code=sm_50 --generate-code arch=compute_52,code=sm_52 --relocatable-device-code true -lcudadevrt --compiler-options -w)
endif()

# GEANT4
find_package(Geant4 REQUIRED)
include(${Geant4_USE_FILE})


# GGEMS src
include_directories(
    ${PROJECT_SOURCE_DIR}/detectors
    ${PROJECT_SOURCE_DIR}/data
    ${PROJECT_SOURCE_DIR}/geometries
    ${PROJECT_SOURCE_DIR}/global
    ${PROJECT_SOURCE_DIR}/tools
    ${PROJECT_SOURCE_DIR}/navigators
    ${PROJECT_SOURCE_DIR}/processes
    ${PROJECT_SOURCE_DIR}/sources)

# Locate sources and headers for this project
file(GLOB sourcesGPU
    ${PROJECT_SOURCE_DIR}/detectors/*.cu
    ${PROJECT_SOURCE_DIR}/data/*.cu
    ${PROJECT_SOURCE_DIR}/geometries/*.cu
    ${PROJECT_SOURCE_DIR}/global/*.cu
    ${PROJECT_SOURCE_DIR}/tools/*.cu
    ${PROJECT_SOURCE_DIR}/navigators/*.cu
    ${PROJECT_SOURCE_DIR}/processes/*.cu
    ${PROJECT_SOURCE_DIR}/sources/*.cu)

file(GLOB headersGPU
    ${PROJECT_SOURCE_DIR}/detectors/*.cuh
    ${PROJECT_SOURCE_DIR}/data/*.cuh
    ${PROJECT_SOURCE_DIR}/geometries/*.cuh
    ${PROJECT_SOURCE_DIR}/global/*.cuh
    ${PROJECT_SOURCE_DIR}/tools/*.cuh
    ${PROJECT_SOURCE_DIR}/navigators/*.cuh
    ${PROJECT_SOURCE_DIR}/processes/*.cuh
    ${PROJECT_SOURCE_DIR}/sources/*.cuh)

file(GLOB ggemssh "ggems.sh")
file(GLOB datafiles "data/*")

# Compilation
find_package(CUDA REQUIRED)
set(LIB_NAME "ggems")
set(CUDA_NVCC_FLAGS ${LIBSCUDA})

SET( CUDA_PROPAGATE_HOST_FLAGS OFF CACHE BOOL "" FORCE )

cuda_add_library(${LIB_NAME} STATIC ${sourcesGPU} ${headersGPU})

# Installation
install(TARGETS ${LIB_NAME} DESTINATION "lib")
install(FILES ${headersGPU} DESTINATION "include")
install(FILES ${ggemssh} DESTINATION "bin" PERMISSIONS WORLD_EXECUTE WORLD_READ OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE)
install(FILES ${datafiles} DESTINATION "data")

install(CODE "MESSAGE(\"Installation done ...\")")
if (UNIX)
install(CODE "MESSAGE(\"To use GGEMS, copy this line in your .bashrc : \")")
install(CODE "MESSAGE(\"    source ${CMAKE_INSTALL_PREFIX}/bin/ggems.sh \")")
endif(UNIX)
if (APPLE)
install(CODE "MESSAGE(\" TO BE CHANGED FOR APPLE CONFIGURATIONS \")")
install(CODE "MESSAGE(\"To use GGEMS, copy this line in your .bashrc : \")")
install(CODE "MESSAGE(\"    source ${CMAKE_INSTALL_PREFIX}/bin/ggems.sh \")")
endif(APPLE)
