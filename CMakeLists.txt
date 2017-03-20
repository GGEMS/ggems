#----------------------------------------------------------------------------
# Setup the project
#
cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

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

# Configuration
find_package(CUDA REQUIRED)
set(BUILD_SHARED_LIBS OFF)
set(CUDA_SEPARABLE_COMPILATION ON)

# OPTIONS SINGLE PRECISION
option(WITH_DOUBLE_PRECISION "Float numbers can be computed in double precision" ON)
if(WITH_DOUBLE_PRECISION)
  #add_definitions(-DSINGLE_PRECISION=0)
else()
  add_definitions(-DSINGLE_PRECISION)
endif()

# CUDA VERSION
message(STATUS "Your CUDA version is ${CUDA_VERSION_MAJOR}.")
if(CUDA_VERSION_MAJOR LESS 7)
    message(FATAL_ERROR "Cuda version should be >= 7.0")
endif()
    
# CUDA OPTIONS    
set(LIBSCUDA --relocatable-device-code true -lcudadevrt --compiler-options -w)   

option(WITH_ARCH_30 "With arch and sm 3.0" ON)
if(WITH_ARCH_30)
    set(LIBSCUDA ${LIBSCUDA} --generate-code arch=compute_30,code=sm_30)
endif()

option(WITH_ARCH_32 "With arch and sm 3.2" ON)
if(WITH_ARCH_32)
    set(LIBSCUDA ${LIBSCUDA} --generate-code arch=compute_32,code=sm_32)
endif()

option(WITH_ARCH_35 "With arch and sm 3.5" ON)
if(WITH_ARCH_35)
    set(LIBSCUDA ${LIBSCUDA} --generate-code arch=compute_35,code=sm_35)
endif()

option(WITH_ARCH_37 "With arch and sm 3.7" ON)
if(WITH_ARCH_37)
    set(LIBSCUDA ${LIBSCUDA} --generate-code arch=compute_37,code=sm_37)
endif()

option(WITH_ARCH_50 "With arch and sm 5.0" ON)
if(WITH_ARCH_50)
    set(LIBSCUDA ${LIBSCUDA} --generate-code arch=compute_50,code=sm_50)
endif()

option(WITH_ARCH_52 "With arch and sm 5.2" ON)
if(WITH_ARCH_52)
    set(LIBSCUDA ${LIBSCUDA} --generate-code arch=compute_52,code=sm_52)
endif()

option(WITH_ARCH_60 "With arch and sm 6.0" ON)
if(WITH_ARCH_60)
    set(LIBSCUDA ${LIBSCUDA} --generate-code arch=compute_60,code=sm_60)
endif()

option(WITH_ARCH_61 "With arch and sm 6.1" ON)
if(WITH_ARCH_61)
    set(LIBSCUDA ${LIBSCUDA} --generate-code arch=compute_61,code=sm_61)
endif()

option(WITH_ARCH_62 "With arch and sm 6.2" ON)
if(WITH_ARCH_62)
    set(LIBSCUDA ${LIBSCUDA} --generate-code arch=compute_62,code=sm_62)
endif()

# OPTIONS FAST MATH
option(WITH_FAST_MATH "Fast math flag" ON)
if(WITH_FAST_MATH)
    set(LIBSCUDA ${LIBSCUDA} -use_fast_math)
endif()

# OPTIONS VERBOSE
option(WITH_KERNEL_VERBOSE "Compilation kernel verbose flag" OFF)
if(WITH_KERNEL_VERBOSE)
    set(LIBSCUDA ${LIBSCUDA} --ptxas-options=-v)
endif()

# GGEMS src
include_directories(
    ${PROJECT_SOURCE_DIR}/alphadev
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
    ${PROJECT_SOURCE_DIR}/alphadev/vox_phan_vrt_nav.cu
    ${PROJECT_SOURCE_DIR}/detectors/*.cu
    ${PROJECT_SOURCE_DIR}/data/*.cu
    ${PROJECT_SOURCE_DIR}/geometries/*.cu
    ${PROJECT_SOURCE_DIR}/global/*.cu
    ${PROJECT_SOURCE_DIR}/tools/*.cu
    ${PROJECT_SOURCE_DIR}/navigators/*.cu
    ${PROJECT_SOURCE_DIR}/processes/*.cu
    ${PROJECT_SOURCE_DIR}/sources/*.cu)

file(GLOB headersGPU
    ${PROJECT_SOURCE_DIR}/aplhadev/vox_phan_vrt_nav.cuh
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
set(LIB_NAME "ggems")
set(CUDA_NVCC_FLAGS ${LIBSCUDA})
#message(${CUDA_NVCC_FLAGS})
SET( CUDA_PROPAGATE_HOST_FLAGS OFF CACHE BOOL "" FORCE )

set(CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE} "--std=c++11")
set(CUDA_NVCC_FLAGS_RELEASE ${CUDA_NVCC_FLAGS_RELEASE} "--std=c++11")

cuda_add_library(${LIB_NAME} STATIC ${sourcesGPU} ${headersGPU})

# Installation
set(TEMPLATE 
    ${PROJECT_SOURCE_DIR}/sources/point_source.cu
    ${PROJECT_SOURCE_DIR}/sources/point_source.cuh
    ${PROJECT_SOURCE_DIR}/detectors/template_detector.cu
    ${PROJECT_SOURCE_DIR}/detectors/template_detector.cuh)

install(TARGETS ${LIB_NAME} DESTINATION "lib")
install(FILES ${headersGPU} DESTINATION "include")
install(FILES ${ggemssh} DESTINATION "bin" PERMISSIONS WORLD_EXECUTE WORLD_READ OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE)
install(FILES ${datafiles} DESTINATION "data")
install(FILES ${TEMPLATE} DESTINATION "dev-template")

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
